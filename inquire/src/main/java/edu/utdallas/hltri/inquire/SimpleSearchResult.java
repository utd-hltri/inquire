package edu.utdallas.hltri.inquire;

public class SimpleSearchResult<T> implements SearchResult<T> {
  private final int rank;
  private final double score;
  private final T value;

  public SimpleSearchResult(int rank, double score, T value) {
    this.rank = rank;
    this.score = score;
    this.value = value;
  }

  @Override
  public int getRank() {
    return rank;
  }

  @Override
  public double getScore() {
    return score;
  }

  @Override
  public T getValue() {
    return value;
  }
}
