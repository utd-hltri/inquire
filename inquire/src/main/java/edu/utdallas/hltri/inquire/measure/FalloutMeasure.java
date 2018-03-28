package edu.utdallas.hltri.inquire.measure;

import com.google.common.collect.Sets;

import java.util.Set;

@SuppressWarnings("unused")
public class FalloutMeasure<T> implements Measure<T> {
  @Override
  public double apply(Iterable<? extends T> retrieved, Iterable<? extends T> relevant) {
    final Set<? extends T> retrievedSet = Sets.newHashSet(retrieved),
                           nonRelevantSet = Sets.difference(retrievedSet, Sets.newHashSet(relevant));
    return Sets.intersection(nonRelevantSet, retrievedSet).size() / (double) nonRelevantSet.size();
  }
}
