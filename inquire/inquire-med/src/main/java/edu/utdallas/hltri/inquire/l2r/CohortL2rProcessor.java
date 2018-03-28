package edu.utdallas.hltri.inquire.l2r;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.googlecode.concurrenttrees.radix.node.concrete.SmartArrayBasedNodeFactory;
import com.googlecode.concurrenttrees.radixinverted.ConcurrentInvertedRadixTree;
import edu.utdallas.hlt.medbase.snomed.SNOMEDManager;
import edu.utdallas.hlt.medbase.snomed.SNOMEDRelationshipDirection;
import edu.utdallas.hlt.medbase.snomed.SNOMEDRelationshipType;
import edu.utdallas.hlt.medbase.umls.UMLSManager;
import edu.utdallas.hltri.inquire.eval.QRels;
import edu.utdallas.hltri.inquire.lucene.HasLuceneId;
import edu.utdallas.hltri.inquire.lucene.LuceneSearchEngine;
import edu.utdallas.hltri.inquire.lucene.similarity.Similarities;
import edu.utdallas.hltri.inquire.text.Keyword;
import edu.utdallas.hltri.knowledge.WikiRedirectManager;
import edu.utdallas.hltri.logging.Logger;
import edu.utdallas.hltri.metamap.MetamapCandidate;
import edu.utdallas.hltri.metamap.MetamapConcept;
import edu.utdallas.hltri.ml.Extractors;
import edu.utdallas.hltri.ml.Feature;
import edu.utdallas.hltri.ml.MultiFeature;
import edu.utdallas.hltri.ml.NumericFeature;
import edu.utdallas.hltri.scribe.text.BaseDocument;
import edu.utdallas.hltri.scribe.text.Document;
import edu.utdallas.hltri.scribe.text.Identifiable;
import edu.utdallas.hltri.scribe.text.annotation.MedicalConcept;
import edu.utdallas.hltri.scribe.text.annotation.NegationSpan;
import edu.utdallas.hltri.scribe.text.annotation.Token;
import edu.utdallas.hltri.struct.Triple;
import edu.utdallas.hltri.struct.Weighted;
import edu.utdallas.hltri.util.Expander;
import edu.utdallas.hltri.util.IntIdentifier;
import gnu.trove.map.TIntFloatMap;
import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.ToIntBiFunction;
import java.util.stream.Collectors;
import javax.annotation.Nonnull;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.similarities.Similarity;

/**
 * Created by travis on 11/2/16.
 */
public class CohortL2rProcessor<
    Topic extends BaseDocument,
    Patient extends Identifiable,
    Report extends HasLuceneId> {
  private final static Logger log = Logger.get(CohortL2rProcessor.class);

  static {
    BooleanQuery.setMaxClauseCount(8192);
  }


  private final Multimap<String, Triple<String, String, String>> topicQueryKeys =
      ArrayListMultimap.create();

  // Enumerate similarity measures we are using
  private final static Map<String, Similarity> similarities =
      new ImmutableMap.Builder<String, Similarity>()
          .put("tf-idf", Similarities.TFIDF.similarity)
          .put("BM25", Similarities.BM25.similarity)
          .put("lm.JM", Similarities.LMJM.similarity)
          .put("lm.Dir", Similarities.LMD.similarity)
          .build();

  /**
   * Calculates sum, min, max, average, variance
   * @param collectionFeature Feature containing list to gather statistics about
   * @return Collection of features of the form (collectionFeature.name.statistic -> statistic)
   */
  public static Collection<Feature<Number>> getStatistics(
      Feature<? extends Collection<Number>> collectionFeature) {
    return NumericFeature.getStatistics(collectionFeature);
  }

  private final List<LoadingCache<CharSequence, Collection<String>>> caches = new ArrayList<>();

  private Expander<CharSequence, String> cachingExpander(
      Expander<CharSequence, String> expander) {
    final LoadingCache<CharSequence, Collection<String>> map = CacheBuilder.newBuilder()
        .concurrencyLevel(Runtime.getRuntime().availableProcessors())
        .build(
            new CacheLoader<CharSequence, Collection<String>>() {
              @Override
              public Collection<String> load(@Nonnull CharSequence key) throws Exception {
                return expander.expand(key);
              }
            }
        );
    caches.add(map);
    return Expander.fromFunction(expander.getName(),
        (CharSequence keyphrase) -> map.getUnchecked(keyphrase.toString()));
  }

  private final SmartArrayBasedNodeFactory factory = new SmartArrayBasedNodeFactory();

  private class BasicQuery {
    private final Set<String> positiveAspects;
    private final Set<String> negativeAspects;

    final Collection<Weighted<String>> positiveKeyphrases = new ArrayList<>();
    final Collection<Weighted<String>> negativeKeyphrases = new ArrayList<>();

    BasicQuery(Collection<? extends CharSequence> aspects) {
      this(aspects, x -> false);
    }

    <E extends CharSequence> BasicQuery(Collection<E> aspects,
               Function<E, Boolean> negatedTest) {

      final Map<Boolean, Set<E>> concepts = aspects.stream()
          .collect(Collectors.groupingBy(negatedTest, Collectors.toSet()));
      this.positiveAspects = Sets.newHashSet(
          Iterables.transform(concepts.getOrDefault(false, Collections.emptySet()),
              CharSequence::toString));
      this.negativeAspects = Sets.newHashSet(
          Iterables.transform(concepts.getOrDefault(true, Collections.emptySet()),
              CharSequence::toString));
    }

    ConcurrentMap<String, Set<String>> expansions = new ConcurrentHashMap<>();

    String printExpansions() {
      return positiveKeyphrases.toString() + negativeKeyphrases.toString();
    }


    int getNumberOfComponents() {
      return positiveAspects.size() + negativeAspects.size();
    }

    BasicQuery expand(Expander<CharSequence, String> expander) {


      ConcurrentMap<String, ConcurrentInvertedRadixTree<String>> expansionTree =
          new ConcurrentHashMap<>();

      for (String keyphrase : Iterables.concat(positiveAspects, negativeAspects)) {
        final Collection<String> newExpansions = expander.expand(keyphrase);
        if (!expansions.containsKey(keyphrase)) {
          expansionTree.put(keyphrase, new ConcurrentInvertedRadixTree<>(factory));
          expansions.put(keyphrase, new HashSet<>());
        }

        for (String expansion : newExpansions) {
          if (!expansionTree.get(keyphrase).getKeysContainedIn(expansion).iterator().hasNext()) {
            expansions.get(keyphrase).add(expansion);
            expansionTree.get(keyphrase).put(expansion, expansion);
          }
        }
      }

      final ConcurrentInvertedRadixTree<String> terms = new ConcurrentInvertedRadixTree<>(factory);

      for (String term : positiveAspects) {
        if (!terms.getKeysContainedIn(term).iterator().hasNext()) {
          positiveKeyphrases.add(Weighted.create(2.0, term));
          terms.put(term, term);

          for (String exp : expansions.get(term)) {
            if (!terms.getKeysContainedIn(exp).iterator().hasNext()) {
              positiveKeyphrases.add(Weighted.create(0.5, exp));
              terms.put(term, term);
            }
          }
        }
      }

      for (String term : negativeAspects) {
        if (!terms.getKeysContainedIn(term).iterator().hasNext()) {
          negativeKeyphrases.add(Weighted.create(-2.0, term));
          terms.put(term, term);

          for (String exp : expansions.get(term)) {
            if (!terms.getKeysContainedIn(exp).iterator().hasNext()) {
              negativeKeyphrases.add(Weighted.create(-0.5, exp));
              terms.put(term, term);
            }
          }
        }
      }

      return this;
    }

    Query toQuery(String stream) {
      BooleanQuery query = searcher.newBooleanQuery(positiveKeyphrases, stream);
      if (!negativeAspects.isEmpty()) {
        BooleanQuery.Builder parent = new BooleanQuery.Builder();
        parent.add(query, BooleanClause.Occur.SHOULD);
        parent.add(searcher.newBooleanQuery(negativeKeyphrases, stream),
            BooleanClause.Occur.SHOULD);
        query = parent.build();
      }
      return query;
    }

    int getNumberOfExpandedComponents() {
      return positiveKeyphrases.size() + negativeKeyphrases.size();
    }
  }

  private final LuceneSearchEngine<?> searcher;
  private final String[] streams;

  public CohortL2rProcessor(LuceneSearchEngine<?> searcher, String[] streams) {
    this.searcher = searcher;
    this.streams = streams;
  }

  private void expand(List<String> expanders,
                      Map<Triple<String, String, String>, BasicQuery> queries,
                      List<Document<Topic>> topics,
                      Map<String, Function<Document<Topic>, BasicQuery>> queryRepresentations,
                      Supplier<Expander<CharSequence, String>> expanderSupplier,
                      String name) {
    final Expander<CharSequence, String> rootExpander = expanderSupplier.get();
    final Expander<CharSequence, String> cachingUmls = cachingExpander(rootExpander);
    expanders.add(name);
    for (Document<Topic> topic : topics) {
      for (Entry<String, Function<Document<Topic>, BasicQuery>> qc :
          queryRepresentations.entrySet()) {
        final BasicQuery query = qc.getValue().apply(topic).expand(cachingUmls);
        log.info("Expanded {}.{} by {} to {}", topic.getId(), qc.getKey(),
            name, query.printExpansions());
        final Triple<String, String, String> key = Triple.of(topic.getId(), qc.getKey(), name);
        queries.put(key, query);
        topicQueryKeys.put(topic.getId(), key);
      }
    }
    if (rootExpander instanceof Closeable) {
      try {
        ((Closeable) rootExpander).close();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  public void process(List<Document<Topic>> topics,
                      Function<String, Collection<Patient>> queryHits,
                      QRels qrels,
                      Function<Patient, Collection<Report>> reportMapper,
                      Path targetPath) {
    log.info("Processing {} topics", topics.size());

    // Enumerate the query representations we are using
    final Map<String, Function<Document<Topic>, BasicQuery>> queryRepresentations =
        new HashMap<>();

    queryRepresentations.put("QC[1]:MetaMap", topic ->
        new BasicQuery(topic.get("metamap", MetamapConcept.TYPE)
            .stream()
            .map(MetamapConcept::getBestCandidate)
            .collect(Collectors.toList())));

    queryRepresentations.put("QC[2]:MetaMapNeg", topic ->
        new BasicQuery(topic.get("metamap", MetamapConcept.TYPE)
            .stream()
            .map(MetamapConcept::getBestCandidate).collect(Collectors.toList()),
            MetamapCandidate::isNegated));

    queryRepresentations.put("QC[3]:Concepts", topic ->
        new BasicQuery(topic.get("kirk", MedicalConcept.from2010)));

    queryRepresentations.put("QC[4]:ConceptsNeg", topic ->
        new BasicQuery(topic.get("kirk", MedicalConcept.from2010),
            mc -> mc.get(MedicalConcept.ASSERTION).equals("ABSENT")));

    queryRepresentations.put("QC[4]:WikiTitles", topic ->
        new BasicQuery(topic.get("wiki", Keyword.TYPE)));

    queryRepresentations.put("QC[5]:WikiTitlesNeg", topic ->
        new BasicQuery(topic.get("wiki", Keyword.TYPE),
            kw -> !kw.getCovering("lingscope", NegationSpan.TYPE).isEmpty()));

    queryRepresentations.put("QC[6]:BoW", topic ->
        new BasicQuery(topic.get("genia", Token.TYPE)));

    queryRepresentations.put("QC[7]:BoWNeg", topic ->
        new BasicQuery(topic.get("genia", Token.TYPE),
            kw -> !kw.getCovering("lingscope", NegationSpan.TYPE).isEmpty()));

    final List<String> expanders = new ArrayList<>();
    // Topic -> QC -> QE
    final Map<Triple<String, String, String>, BasicQuery> queries = new HashMap<>();


    // Open all the resources we will be using
    expand(expanders,queries, topics, queryRepresentations, UMLSManager::new, "QE[1]:UMLS");
    try (SNOMEDManager snomed = new SNOMEDManager()) {
      expand(expanders, queries, topics, queryRepresentations,  () -> snomed.expandBy(
          SNOMEDRelationshipType.IS_A, 2,
          SNOMEDRelationshipDirection.CHILDREN), "QE[2]:UMLS");
    }

    expand(expanders, queries, topics, queryRepresentations,
        WikiRedirectManager::new, "QE[3]:Wiki");
    expand(expanders, queries, topics, queryRepresentations, UMLSManager::new, "QE[4]:PRF");

    for (Document<Topic> topic : topics) {
      for (Entry<String, Function<Document<Topic>, BasicQuery>> qc :
          queryRepresentations.entrySet()) {
        Expander<CharSequence, String> e =
            Expander.fromFunction("ALL",
                (CharSequence cs) -> expanders.stream()
                    .flatMap(qe -> queries.get(Triple.of(topic.getId(), qc.getKey(), qe))
                                          .expansions.get(cs.toString())
                                          .stream())
                    .collect(Collectors.toSet()));
        final Triple<String, String, String> key = Triple.of(topic.getId(), qc.getKey(), "QE[5]:ALL");
        queries.put(key, qc.getValue().apply(topic).expand(e));
        topicQueryKeys.put(topic.getId(), key);
      }
    }

    // Initialize feature sets
    final Collection<Function<Document<Topic>, Collection<? extends Feature<Number>>>> topicFeatures = Lists.newArrayList();
    final Collection<Function<Patient, Collection<? extends Feature<Number>>>> patientFeatures = Lists.newArrayList();
    final Collection<BiFunction<Document<Topic>, Patient, Collection<? extends Feature<Number>>>> relevanceFeatures = Lists.newArrayList();

    // Topic Features
    for (Entry<String, Function<Document<Topic>, BasicQuery>> qc :
        queryRepresentations.entrySet()) {
      // TF1[#Components]
      topicFeatures.add(
          Extractors.singleInt("#components",
              topic -> qc.getValue().apply(topic).getNumberOfComponents()));
      for (String qe : expanders) {
        // TF2[#ExpandedComponents]
        topicFeatures.add(
            Extractors.singleInt("#expanded_components." + qc.getKey() + "." + qe,
                topic -> queries.get(Triple.of(topic.getId(), qc.getKey(), qe))
                    .getNumberOfExpandedComponents()));

        // TF3[n-idf]
        for (String stream : streams) {
          topicFeatures.add(Extractors.singleDouble("nIDF." + qc.getKey() + "." + qe + "." + stream,
              topic -> searcher.getInverseDocumentFrequency(queries.get(Triple.of(topic.getId(), qc.getKey(), qe)).toQuery(stream))));
        }
      }
    }

    // Patient Features
    // PF1[#Records]
    patientFeatures.add(
        Extractors.singleInt("#records",
            patient -> reportMapper.apply(patient).size()));

    // PF2[%ReportTypes]
    patientFeatures.add(Extractors.singleDouble("%record-types*", patient -> 0d));

    for (String stream : streams) {
      // PF3[StreamLength]
      patientFeatures.add(patient -> {
        final List<Feature<Number>> lengths = Lists.newArrayList();
        for (final Report report : reportMapper.apply(patient)) {
          lengths.add(Feature.numericFeature("length." + stream,
              searcher.getFieldLength(report.getLuceneId(), stream)));
        }
        return getStatistics(MultiFeature.flatten(lengths));
      });

    }

    // Dynamic Features
    // Age score
    relevanceFeatures.add(Extractors.singleDouble("filter.age*", (t, p) -> 0d));

    // Gender score
    relevanceFeatures.add(Extractors.singleDouble("filter.gender*", (t, p) -> 0d));

    // Status score
    relevanceFeatures.add(Extractors.singleDouble("filter.status*", (t, p) -> 0d));

    ConcurrentMap<String, TIntFloatMap> cachedScores = new ConcurrentHashMap<>();

    for (Entry<String, Function<Document<Topic>, BasicQuery>> qc :
        queryRepresentations.entrySet()) {
      for (String qe : expanders) {
        for (String stream : streams) {
          for (Entry<String, Similarity> sim : similarities.entrySet()) {
            // (Topic - Patient) similarity score
            relevanceFeatures.add((topic, patient) -> {
              final List<Feature<Number>> scores = Lists.newArrayList();

              final String name = "relevance." + sim.getKey() + "." + qc.getKey() + '.' +
                  qe + '.' + stream;

              final String key = topic.getId() + '.' + qc.getKey() + '.' + qe + '.' + stream + '.' + sim.getKey();

              cachedScores.computeIfAbsent(key,
                  x -> searcher.getSimilarities(sim.getValue(), queries.get(Triple.of(topic.getId(), qc.getKey(), qe))
                        .toQuery(stream)));

              for (final Report report : reportMapper.apply(patient)) {
                final double score = cachedScores.get(key).get(report.getLuceneId());
                assert !Double.isNaN(score) && Double.isFinite(score) && Double.compare(score, 0d) > 0 :
                    "Got score of {} for " + name;
                scores.add(Feature.numericFeature(name, score));
              }
              return getStatistics(MultiFeature.flatten(scores));
            });

            /*
            relevanceFeatures.add((topic, patient) -> {
              final List<Feature<Number>> scores = Lists.newArrayList();
              for (final Report report : reportMapper.apply(patient)) {
                final String name = "relevance." + sim.getKey() + "." + qc.getKey() + '.' +
                    qe.getKey() + '.' + stream;
                scores.add(Feature.numericFeature(name, searcher
                    .getNormalizedTermFrequency(sim.getValue(),
                        qc.getValue().apply(topic)
                            .expand(qe.getValue())
                            .toQuery(stream),
                        report.getLuceneId(),
                        stream)));
              }
              return getStatistics(MultiFeature.flatten(scores));
            });
            */
          }
        }
      }
    }

    final ToIntBiFunction<Document<Topic>, Patient> judgementFunction =
        (topic, patient) -> qrels.getRelevance(topic.getId(), patient.getId()).toInt();

    // Do the actual vectorization!
    final RankingFeatureExtractor<Document<Topic>, Patient> extractor =
        new RankingFeatureExtractor<>(
        topicFeatures,
        patientFeatures,
        relevanceFeatures,
        judgementFunction);

    extractor.vectorize(
        new IntIdentifier<>(),
        topics,
        topic -> queryHits.apply(topic.getId()),
        topic -> {
          topicQueryKeys.get(topic.getId()).forEach(queries::remove);
          caches.forEach(LoadingCache::invalidateAll);
          cachedScores.clear();
        },
        targetPath.resolve("vectors.svmr"),
        targetPath.resolve("feature_mapping.tsv"),
        targetPath.resolve("query_mapping.tsv"));

    log.info("Processed {} topics to {}", topics.size(), targetPath);
  }
}
