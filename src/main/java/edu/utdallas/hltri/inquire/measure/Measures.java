package edu.utdallas.hltri.inquire.measure;

import com.google.common.base.Function;
import scala.Tuple2;

public class Measures {
  <T> Function<Tuple2<? extends Iterable<? extends T>, ? extends Iterable<? extends T>>, Double> asFunction(final Measure<T> measure) {
    return new Function<Tuple2<? extends Iterable<? extends T>, ? extends Iterable<? extends T>>, Double>() {
      @Override public Double apply(Tuple2<? extends Iterable<? extends T>, ? extends Iterable<? extends T>> input) {
        return measure.apply(input._1(), input._2());
      }
    };
  }
}
