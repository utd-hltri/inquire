package edu.utdallas.hltri.inquire.l2r;

import com.google.common.base.Preconditions;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.Lists;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.ToIntBiFunction;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import edu.utdallas.hltri.framework.ProgressLogger;
import edu.utdallas.hltri.inquire.eval.QRels;
import edu.utdallas.hltri.inquire.eval.QRels.Relevance;
import edu.utdallas.hltri.logging.Logger;
import edu.utdallas.hltri.ml.Feature;
import edu.utdallas.hltri.ml.vector.SparseFeatureVector;
import edu.utdallas.hltri.ml.vector.SparseFeatureVectorizer;
import edu.utdallas.hltri.scribe.text.Identifiable;
import edu.utdallas.hltri.util.IntIdentifier;

/**
 * Vectorizer for JAMIA 1
 * Created by travis on 8/5/14.
 */
@SuppressWarnings({"WeakerAccess", "Convert2streamapi"})
public class RankingFeatureExtractor<Topic extends Identifiable, RetrievedDocument extends Identifiable> {
  private final static Logger log = Logger.get(RankingFeatureExtractor.class);

  private final Collection<Function<Topic, Collection<? extends Feature<Number>>>>
      topicFunctions;

  private final Collection<Function<RetrievedDocument, Collection<? extends Feature<Number>>>>
      documentFunction;

  private final Collection<BiFunction<Topic, RetrievedDocument, Collection<? extends Feature<Number>>>>
      dynamicFunctions;

  private final ToIntBiFunction<Topic, RetrievedDocument>
      judgementFunction;

  public RankingFeatureExtractor(
      final Collection<Function<Topic, Collection<? extends Feature<Number>>>> topicFunctions,
      final Collection<Function<RetrievedDocument, Collection<? extends Feature<Number>>>> visitFunctions,
      final Collection<BiFunction<Topic, RetrievedDocument, Collection<? extends Feature<Number>>>> dynamicFunctions,
      final ToIntBiFunction<Topic, RetrievedDocument> judgementFunction) {

    this.topicFunctions = topicFunctions;
    log.debug("Configured {} topic functions", topicFunctions.size());
    this.documentFunction = visitFunctions;
    log.debug("Configured {} visit functions", visitFunctions.size());
    this.dynamicFunctions = dynamicFunctions;
    log.debug("Configured {} dynamic functions", dynamicFunctions.size());

    this.judgementFunction = judgementFunction;
    log.info("Initialized Learning-to-Rank Feature Extractor!");
  }

  IntIdentifier<String> topicIdentifier = null;

  public void vectorize(
      final IntIdentifier<String> featureIdentifier,
      final Collection<Topic> topics,
      final Function<Topic, Collection<RetrievedDocument>> documentSupplier,
      final Consumer<Topic> topicCleanup,
      final Path vectorPath,
      final Path featureMappingPath,
      final Path topicMappingPath) {

    log.info("Starting vectorization of {} topics...", topics.size());
    topicIdentifier = new IntIdentifier<>();
    final SparseFeatureVectorizer<Number> sparseVectorizer =
        new SparseFeatureVectorizer<>(featureIdentifier);

    final LoadingCache<RetrievedDocument, List<Feature<Number>>> visitFeatureCache =
        CacheBuilder.newBuilder()
        .maximumSize(1000)
        .build(CacheLoader.from(retrievedDocument -> documentFunction.stream().flatMap(fe ->
            fe.apply(retrievedDocument).stream()).collect(Collectors.toList())));

    final List<L2rFeatureVector> vectors = new ArrayList<>();
    final List<L2rFeatureVector> topicVectors = new ArrayList<>();
    try (final ProgressLogger qlog = ProgressLogger.fixedSize("Processing Topics", topics.size(),
        1, TimeUnit.SECONDS)) {
      // Iterate across topics
      for (Topic topic : topics) {
        // Enumerate topic features
        final List<Feature<Number>> topicFeatures = Lists.newArrayList();
        topicFunctions.forEach(extractor -> topicFeatures.addAll(extractor.apply(topic)));

        final Collection<RetrievedDocument> retrievedDocuments = documentSupplier.apply(topic);
        final AtomicInteger i = new AtomicInteger();
        for (RetrievedDocument retrievedDocument : retrievedDocuments) {
          i.incrementAndGet();

          // Get and cache retrievedDocument features if needed
          final Stream<Feature<Number>> documentFeatures =
              visitFeatureCache.getUnchecked(retrievedDocument).stream();

          // Enumerate dynamic features
          final Stream<Feature<Number>> dynamicFeatures = dynamicFunctions.stream().parallel()
              .flatMap(fe -> fe.apply(topic, retrievedDocument).stream());

          final SparseFeatureVector<Number> sparseVector = sparseVectorizer.vectorize(
              Stream.of(topicFeatures.stream(),
                  dynamicFeatures,
                  documentFeatures)
                  .flatMap(Function.identity()));

          final int topicId = topicIdentifier.getIDOrAdd(topic.getId());
          final int judgment = judgementFunction.applyAsInt(topic, retrievedDocument);

          topicVectors.add(new L2rFeatureVector(topicId, judgment, sparseVector, topic.getId() + "::" + retrievedDocument.getId()));
          if (retrievedDocument instanceof Closeable) {
            ((Closeable) retrievedDocument).close();
          }
        }

        Collections.shuffle(topicVectors, ThreadLocalRandom.current());
        vectors.addAll(topicVectors);
        topicVectors.clear();
        qlog.update("Generated {} feature vectors for topic {}", i.get(), topic.getId());
        topicCleanup.accept(topic);
      }

      try (BufferedWriter writer = Files.newBufferedWriter(vectorPath)) {
        for (L2rFeatureVector vector : vectors) {
          writer.append(vector.makeDense().toSvmRankFormat());
          writer.newLine();
        }
      }

      featureIdentifier.toFile(featureMappingPath, true);
      topicIdentifier.toFile(topicMappingPath);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public void saveMappedQrels(QRels original, Path outputQrels) {
    Preconditions.checkState(topicIdentifier != null, "must call vectorize() first!");

    try (BufferedWriter writer = Files.newBufferedWriter(outputQrels)) {
      for (final String originalTopic : original.getTopics()) {
        final String mappedTopic = Integer.toString(topicIdentifier.getID(originalTopic));
        for (Entry<String, Relevance> entry : original.getJudgements(originalTopic).entrySet()) {
          final String docId = entry.getKey();
          final String relevance = Integer.toString(entry.getValue().toInt());
          writer.append(mappedTopic).append(" 0 ").append(docId).append(' ').append(relevance);
          writer.newLine();
        }
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

  }
}
