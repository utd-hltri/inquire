package edu.utdallas.hltri.inquire.engines;

import edu.utdallas.hltri.inquire.SearchResult;

public interface SearchEngine<Q, D, R extends SearchResult<D>> {
  SearchResults<D, R> search(final Q query, final int numResults);
}
