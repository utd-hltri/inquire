package edu.utdallas.hltri.inquire.measure;

@SuppressWarnings("unused")
public class SpecificityMeasure<T> implements Measure<T> {
  private final Measure<T> fallout = new FalloutMeasure<>();

  @Override
  public double apply(Iterable<? extends T> retrieved, Iterable<? extends T> relevant) {
    return 1 - fallout.apply(retrieved, relevant);
  }
}
