package edu.utdallas.hltri.inquire.lucene;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;

/**
 * Created by travis on 7/14/14.
 */
@SuppressWarnings("unused")
public class EmptyResult extends LuceneResult<Object> {
  public final static EmptyResult INSTANCE = new EmptyResult();

  private EmptyResult() {
    super(new Object(), 0.0, 0, -1);
  }

  public static class Factory implements DocumentFactory<EmptyResult> {
    @Override public EmptyResult build(IndexReader reader, int luceneId) {
      return INSTANCE;
    }
  }
}
