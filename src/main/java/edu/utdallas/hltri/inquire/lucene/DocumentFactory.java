package edu.utdallas.hltri.inquire.lucene;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;

import java.io.IOException;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Created by travis on 5/31/17.
 */
@FunctionalInterface
public interface DocumentFactory<T> {

  T build(IndexReader reader, int luceneId);

  static <T> DocumentFactory<T> eager(Function<Document, T> factory) {
    return (reader, luceneId) -> {
      try {
        return factory.apply(reader.document(luceneId));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    };
  }
}
