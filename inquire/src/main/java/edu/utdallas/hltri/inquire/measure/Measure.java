package edu.utdallas.hltri.inquire.measure;

public interface Measure<T>  {
  double apply(Iterable<? extends T> retrieved, Iterable<? extends T> relevant);
}
