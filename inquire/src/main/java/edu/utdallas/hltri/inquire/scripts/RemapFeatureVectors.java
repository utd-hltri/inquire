package edu.utdallas.hltri.inquire.scripts;

import com.google.common.base.CharMatcher;
import com.google.common.base.Splitter;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import edu.utdallas.hltri.framework.Commands;
import edu.utdallas.hltri.inquire.l2r.L2rFeatureVector;
import edu.utdallas.hltri.ml.Feature;
import edu.utdallas.hltri.ml.vector.SparseFeatureVectorizer;
import edu.utdallas.hltri.util.IntIdentifier;
import edu.utdallas.hltri.util.Unsafe;
import picocli.CommandLine.Parameters;

public class RemapFeatureVectors implements Runnable {
  @Parameters(index = "0", paramLabel = "INPUT-VECTORS")
  Path inputVectorsFile;

  @Parameters(index = "1", paramLabel = "INPUT-FEATURE-MAP")
  Path inputFeatureMapFile;

  @Parameters(index = "2", paramLabel = "TARGET-FEATURE-MAP")
  Path targetFeatureMapFile;

  @Parameters(index = "3", paramLabel = "OUTPUT-VECTORS")
  Path outputVectorsFile;

  @Override
  public void run() {
    final IntIdentifier<String> inputFeatureMap = IntIdentifier.fromFile(inputFeatureMapFile).lock();
    final IntIdentifier<String> targetFeatureMap = IntIdentifier.fromFile(targetFeatureMapFile).lock();

    final SparseFeatureVectorizer<Number> vectorizer = new SparseFeatureVectorizer<>(targetFeatureMap);
    final List<L2rFeatureVector> vectors = new ArrayList<>();

    final Splitter splitter = Splitter.on(CharMatcher.whitespace()).omitEmptyStrings().trimResults();
    try {
      for (String line : Files.readAllLines(inputVectorsFile)) {
        String comment = null;
        int commentDelim = line.indexOf('#');
        if (commentDelim >= 0) {
          comment = line.substring(commentDelim + 1);
          line = line.substring(0, commentDelim).trim();
        }
        final List<String> entries = splitter.splitToList(line);
        assert entries.size() > 2;
        int label = Integer.valueOf(entries.get(0));
        int topicId = Integer.valueOf(entries.get(1).substring(4));
        final Stream<Feature<Number>> features = entries.stream().skip(2).map(Unsafe.function(
            entry -> {
              int delim = entry.indexOf(':');
              final int inputFeatureId = Integer.parseInt(entry.substring(0, delim));
              return Feature.numericFeature(inputFeatureMap.get(inputFeatureId),
                  Double.valueOf(entry.substring(delim + 1)));
            }
        ));
        vectors.add(new L2rFeatureVector(topicId, label, vectorizer.vectorize(features), comment));
      }
      try (final BufferedWriter writer = Files.newBufferedWriter(outputVectorsFile)) {
        for (L2rFeatureVector vector : vectors) {
          writer.write(vector.makeDense().toSvmRankFormat());
          writer.newLine();
        }
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static void main(String... args) {
    Commands.run(new RemapFeatureVectors(), args);
  }
}
