package edu.utdallas.hltri.inquire;

@SuppressWarnings("unused")
public interface SearchResult<T> {
  T getValue();
  int getRank();
  double getScore();
}
