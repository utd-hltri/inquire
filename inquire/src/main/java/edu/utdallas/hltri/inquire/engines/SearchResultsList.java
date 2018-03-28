package edu.utdallas.hltri.inquire.engines;

import java.util.List;

import edu.utdallas.hltri.inquire.SearchResult;

/**
 * Created by travis on 5/31/17.
 */
public class SearchResultsList<T, R extends SearchResult<T>> implements SearchResults<T, R> {

  @SuppressWarnings("WeakerAccess")
  protected final double  maxScore;

  @SuppressWarnings("WeakerAccess")
  protected final int     numHits;

  @SuppressWarnings("WeakerAccess")
  protected final List<R> resultsList;

  public SearchResultsList(double maxScore, int numHits, List<R> resultsList) {
    this.maxScore = maxScore;
    this.numHits = numHits;
    this.resultsList = resultsList;
  }

  @Override
  public double getMaxScore() {
    return maxScore;
  }

  @Override
  public int getTotalHits() {
    return numHits;
  }

  @Override
  public List<R> getResults() {
    return resultsList;
  }

  public SearchResultsList<T, R> trimToLength(int length) {
    if (length >= resultsList.size()) {
      return this;
    } else {
      return new SearchResultsList<>(maxScore, numHits, resultsList.subList(0, length));
    }
  }
}
