package edu.utdallas.hltri.inquire.measure;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import java.util.List;
import java.util.Set;

@SuppressWarnings("unused")
public class AvgPrecisionMeasure<T> implements Measure<T> {

  final Measure<T> precision = new PrecisionMeasure<>();

  @Override
  public double apply(Iterable<? extends T> retrieved, Iterable<? extends T> relevant) {
    final List<? extends T> retrievedList = Lists.newArrayList(retrieved),
                           relevantList = Lists.newArrayList(relevant);
    final Set<? extends T> relevantSet = Sets.newHashSet(relevant);

    double sum = 0.0;
    for (int i = 0; i < relevantList.size(); i++) {
      if (relevantSet.contains(retrievedList.get(i))) {
        sum += precision.apply(retrievedList.subList(0, i), relevantSet);
      }
    }

    return sum / relevantList.size();
  }
}
