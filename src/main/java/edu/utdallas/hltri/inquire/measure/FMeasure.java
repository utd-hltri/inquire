package edu.utdallas.hltri.inquire.measure;

public class FMeasure<T> implements Measure<T> {

  final double beta;
  final Measure<T> precisionMeasure = new PrecisionMeasure<>(),
                   recallMeasure = new RecallMeasure<>();

  public FMeasure(double beta) {
    this.beta = Math.pow(beta, 2);
  }

  @Override
  public double apply(Iterable<? extends T> retrieved, Iterable<? extends T> relevant) {
    final double precision = precisionMeasure.apply(retrieved, relevant),
                 recall = recallMeasure.apply(retrieved, relevant);
    return (1 + beta) * (precision * recall) / (beta * precision + recall);
  }
}
