package edu.utdallas.hltri.inquire.lucene;

import edu.utdallas.hltri.inquire.SearchResult;

public class LuceneResult<T> implements SearchResult<T> {
  protected double score;
  protected int    rank;

  private final   int luceneId;
  protected T   value;

  public LuceneResult(T value, double score, int rank, int luceneId) {
    this.score = score;
    this.rank = rank;
    this.value = value;
    this.luceneId = luceneId;
  }

  @Override public T getValue() {
    return value;
  }

  @Override public double getScore() {
    return score;
  }

  @Override public int getRank() {
    return rank;
  }

  @SuppressWarnings("unused")
  public int getLuceneDocId() {
    return luceneId;
  }

  public LuceneResult<T> setScore(double score) {
    this.score = score;
    return this;
  }

  @SuppressWarnings("unused")
  public LuceneResult<T> setRank(int rank) {
    this.rank = rank;
    return this;
  }

  @Override public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;

    LuceneResult that = (LuceneResult) o;

    return rank == that.rank && Double.compare(that.score, score) == 0 && value.equals(that.value);
  }

  @Override public int hashCode() {
    int result;
    long temp;
    temp = Double.doubleToLongBits(score);
    result = (int) (temp ^ (temp >>> 32));
    result = 31 * result + rank;
    result = 31 * result + value.hashCode();
    return result;
  }

  void setValue(T value) {
    this.value = value;
  }
}
