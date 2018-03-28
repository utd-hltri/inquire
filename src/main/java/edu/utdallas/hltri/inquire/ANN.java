package edu.utdallas.hltri.inquire;

import com.google.common.base.CharMatcher;
import com.google.common.base.Splitter;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

import edu.utdallas.hltri.io.IOUtils;
import edu.utdallas.hltri.logging.Logger;
import edu.utdallas.hltri.struct.Weighted;

/**
 * Created by travis on 11/30/15.
 */
public class ANN {

  private static final Logger logger = Logger.get(ANN.class);

  static {
    System.out.println(System.getProperty("java.library.path"));
//    System.loadLibrary("fann");
    System.loadLibrary("fannutil");
  }

  public final String[] vocabulary;
  public final Map<String, double[]> vectors = new TreeMap<>();

  public ANN(String data, IndexParams params) {
    this(data, params, Splitter.on(CharMatcher.whitespace()).omitEmptyStrings());
  }

  public ANN(String data, IndexParams params, Splitter splitter) {
    final ArrayList<Double> values = new ArrayList<>();
    String word;

    logger.info("Loading vector mappings...");
    for (final String line : IOUtils.lazy.readLines(data)) {
      final Iterator<String> it = splitter.split(line).iterator();
      word = it.next();
      values.clear();
      while (it.hasNext()) {
        values.add(Double.parseDouble(it.next()));
      }
      vectors.put(word, values.stream().mapToDouble(d -> d).toArray());
    }

    logger.info("Building vocabulary...");
    vocabulary = vectors.keySet().stream().toArray(String[]::new);

    final int N = vectors.values().iterator().next().length;

    double[] concatVector = new double[ N * vectors.size() ];
    int i = 0;
    for (double[] vector : vectors.values()) {
      System.arraycopy(vector, 0, concatVector, N * i, N);
      i++;
    }

    final File index = new File(IOUtils.removeExtension(data) + ".fann.index");

    if (index.exists()) {
      logger.info("Loading ANN indices from {}...", index);
      loadIndex(concatVector, i, N, index.getAbsolutePath());
    } else {
      logger.info("Building ANN indices...");
      params.build(concatVector, i, N);

      logger.info("Saving ANN indices to {}...", index);
      saveIndex(index.getAbsolutePath());
    }
    logger.info("Done!");
  }

  public List<Weighted<String>> getNearest(String word, int n) {
    if (vectors.containsKey(word)) {
      return getNearest(vectors.get(word), n);
    } else {
      return Collections.emptyList();
    }
  }

  public double[] getVector(String vectorId) {
    return vectors.get(vectorId);
  }

  public boolean containsVector(String vectorId) {
    return vectors.containsKey(vectorId);
  }

  public List<Weighted<String>> getNearest(double[] vector, int n) {
    int[] indices = new int[ n ];
    double[] weights = new double[ n ];
    logger.info("Finding {} nearest neighbors to {}", n, Arrays.toString(vector));
    findKNN(vector, 1024, indices, weights);
    List<Weighted<String>> results = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      results.add(Weighted.create(weights[i], vocabulary[indices[i]]));
    }
    Collections.sort(results, Comparator.comparingDouble(Weighted<String>::getWeight));
    return results;
  }


  private static native void findKNN(double[] vector, int checks, int[] indices, double[] weights);

  private static native void saveIndex(String file);
  private static native void loadIndex(double[] data, int rows, int cols, String file);

  protected interface IndexParams {
    void build(double[] data, int rows, int cols);
  }

  public static class LinearIndex implements IndexParams {
    @Override
    public void build(double[] data, int rows, int cols) {
      buildLinearIndex(data, rows, cols);
    }
  }

  public static class KMeansIndex implements IndexParams {
    public final int branching;
    public final int iterations;
    public final double cbIndex;

    public KMeansIndex(int branching, int iterations, double cbIndex) {
      this.branching = branching;
      this.iterations = iterations;
      this.cbIndex = cbIndex;
    }

    @Override
    public void build(double[] data, int rows, int cols) {
      buildKMeansIndex(data, rows, cols, branching, iterations, cbIndex);
    }
  }

  public static class KDTreeIndex implements IndexParams {
    public final int trees;

    public KDTreeIndex(int trees) {
      this.trees = trees;
    }

    @Override
    public void build(double[] data, int rows, int cols) {
      buildKDTreeIndex(data, rows, cols, trees);
    }
  }

  public static class CompositeIndex implements IndexParams {
    public final int trees;
    public final int branching;
    public final int iterations;
    public final double cbIndex;

    public CompositeIndex(int trees, int branching, int iterations, double cbIndex) {
      this.trees = trees;
      this.branching = branching;
      this.iterations = iterations;
      this.cbIndex = cbIndex;
    }

    @Override
    public void build(double[] data, int rows, int cols) {
      buildCompositeIndex(data, rows, cols, trees, branching, iterations, cbIndex);
    }
  }

  public static class LSHIndex implements IndexParams {
    public final int numTables;
    public final int keySize;
    public final int probeLevel;

    public LSHIndex(int numTables, int keySize, int probeLevel) {
      this.numTables = numTables;
      this.keySize = keySize;
      this.probeLevel = probeLevel;
    }

    @Override
    public void build(double[] data, int rows, int cols) {
      buildLSHIndex(data, rows, cols, numTables, keySize, probeLevel);
    }
  }

  public static class AutoIndex implements IndexParams {
    public final double targetPrecision;
    public final double buildWeight;
    public final double memoryWeight;
    public final double sampleFraction;

    public AutoIndex(double targetPrecision, double buildWeight, double memoryWeight,
                     double sampleFraction) {
      this.targetPrecision = targetPrecision;
      this.buildWeight = buildWeight;
      this.memoryWeight = memoryWeight;
      this.sampleFraction = sampleFraction;
    }

    @Override
    public void build(double[] data, int rows, int cols) {
      buildAutoIndex(data, rows, cols, targetPrecision, buildWeight, memoryWeight, sampleFraction);
    }

    public static AutoIndex defaultParams() {
      return new AutoIndex(0.9, 0.01, 0, 0.1);
    }
  }

  private static native void buildLinearIndex(double[] data, int rows, int cols);
  private static native void buildKDTreeIndex(double[] data, int rows, int cols, int trees);
  private static native void buildKMeansIndex(double[] data, int rows, int cols, int branching, int iterations, double cbIndex);
  private static native void buildCompositeIndex(double[] data, int rows, int cols, int trees, int branching, int iterations, double cbIndex);
  private static native void buildLSHIndex(double[] data, int rows, int cols, int numTables, int keySize, int probeLevel);
  private static native void buildAutoIndex(double[] data, int rows, int cols, double targetPrecision, double buildWeight, double memoryWeight, double sampleFraction);

  public static void main(String... args) {
    IndexParams params = null;
    switch(args[1].toUpperCase()) {
      case "LINEAR":
        params = new LinearIndex();
        break;
      case "KDTREE":
        params = new KDTreeIndex(Integer.parseInt(args[2]));
        break;
      case "KMEANS":
        params = new KMeansIndex(
            Integer.parseInt(args[2]),
            Integer.parseInt(args[3]),
            Double.parseDouble(args[4])
        );
        break;
      case "COMPOSITE":
        params = new CompositeIndex(
            Integer.parseInt(args[2]),
            Integer.parseInt(args[3]),
            Integer.parseInt(args[4]),
            Double.parseDouble(args[5])
        );
        break;
      case "LSH":
        params = new LSHIndex(
            Integer.parseInt(args[2]),
            Integer.parseInt(args[3]),
            Integer.parseInt(args[4])
        );
        break;
      case "AUTO":
        params = new AutoIndex(
            Double.parseDouble(args[2]),
            Double.parseDouble(args[3]),
            Double.parseDouble(args[4]),
            Double.parseDouble(args[5])
        );
        break;
    }

    ANN ann = ("tsv".equals(args[args.length-1])) ? new ANN(args[0], params, Splitter.on('\t').omitEmptyStrings())
        : new ANN(args[0], params);

    System.out.print("Enter term:");
    try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in))) {
      for (String term; (term = reader.readLine()) != null; ) {
        final List<Weighted<String>> nearest = ann.getNearest(term, 10);
        System.out.println(nearest.stream().map(Weighted::toString).collect(Collectors.joining("\n")));
        System.out.println();
      }
      System.out.print("Enter term:");
    } catch (IOException e) {
      e.printStackTrace();
    }
    System.out.println();
  }
}
