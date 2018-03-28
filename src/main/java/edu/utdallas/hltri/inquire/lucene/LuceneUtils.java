package edu.utdallas.hltri.inquire.lucene;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;

import org.apache.commons.lang3.tuple.Triple;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BooleanQuery.Builder;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.spans.SpanBoostQuery;
import org.apache.lucene.search.spans.SpanNearQuery;
import org.apache.lucene.search.spans.SpanQuery;
import org.apache.lucene.search.spans.SpanTermQuery;

import java.io.IOException;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import edu.utdallas.hltri.logging.Logger;
import edu.utdallas.hltri.scribe.text.BaseDocument;
import edu.utdallas.hltri.scribe.text.Document;
import edu.utdallas.hltri.struct.Pair;
import edu.utdallas.hltri.struct.Weighted;
import org.apache.lucene.util.BytesRef;

/**
 *
 * Created by travis on 7/14/14.
 */
@SuppressWarnings({"WeakerAccess", "SameParameterValue", "unused"})
public final class LuceneUtils {
  private static final Logger log = Logger.get(LuceneUtils.class);

  private LuceneUtils() {}

  public static BooleanQuery makeBooleanQuery(final List<Query> clauses,
                                              final Occur occur) {
    Preconditions.checkArgument(!clauses.isEmpty(), "cannot create a boolean query with zero clauses");
    final BooleanQuery.Builder bqb = new BooleanQuery.Builder();
    for (Query clause : clauses) {
      bqb.add(clause, occur);
    }
    return bqb.build();
  }

  public static List<String> tokenize(final CharSequence string,
                                      final String field,
                                      final Analyzer analyzer) {
    final List<String> tokens = new ArrayList<>();
    assert Objects.nonNull(analyzer) : "Analyzer was null!";
    assert Objects.nonNull(field) : "Field was null!";
    assert Objects.nonNull(string) : "String was null!";
    try (final TokenStream stream = analyzer.tokenStream(field, string.toString())) {
      stream.reset();
      while (stream.incrementToken()) {
        tokens.add(stream.getAttribute(CharTermAttribute.class).toString());
      }
    } catch (IOException ex) {
      throw new RuntimeException("Lucene said this should never be thrown!", ex);
    }
    return tokens;
  }

  public static LoadingCache<Pair<LeafReader, String>, BinaryDocValues> cache =
      CacheBuilder.<Pair<LeafReader, String>, BinaryDocValues>newBuilder()
      .maximumSize(2048)
      .softValues()
      .build(
          CacheLoader.from(
              (Pair<LeafReader, String> pair) -> {
                try {
                  return DocValues.getBinary(pair.first(), pair.second());
                } catch (IOException e) {
                  throw new RuntimeException(e);
                }
              }
          )
      );

  public static BytesRef getBinaryDocValue(IndexReader reader, String field, int luceneId) {
    BytesRef bytes = null;
    for (final LeafReaderContext leaf : reader.leaves()) {
      if (luceneId < leaf.docBase) {
        continue;
      }
      int docId = luceneId - leaf.docBase;
      final LeafReader leafReader = leaf.reader();
//      try {
        if (docId >= leafReader.maxDoc()) {
          continue;
        }
        if (bytes != null) {
          log.error("Found multiple blobs for {}::{}::{}", reader, field, luceneId);
        }
        bytes = cache.getUnchecked(Pair.of(leafReader, field)).get(docId);
//      } catch (IOException e) {
//        throw new RuntimeException(e);
//      }
    }
    if (bytes != null) {
      return bytes;
    } else {
      throw new IllegalStateException(
          "Unable to find doc" + luceneId + " in any IndexReader leaf!");
    }
  }

  public static long getVocabularySize(IndexReader index) {
    long size = 0L;
    Fields fields;
    for (final LeafReaderContext leaf : index.leaves()) {
      try (final LeafReader reader = leaf.reader()) {
        fields = reader.fields();
        for (String field : fields) {
          size += fields.terms(field).size();
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
    return size;
  }

  public static <T extends Query> Collector<T, ArrayList<BooleanClause>, BooleanQuery> toBooleanQuery(Occur occur) {
    return Collector.of(
        ArrayList::new,
        (clauses, q) -> clauses.add(new BooleanClause(q, occur)),
        (clauses1, clauses2) -> {
          clauses1.addAll(clauses2);
          return clauses1;
        },
        clauses -> {
          final BooleanQuery.Builder bqb = new BooleanQuery.Builder();
          for (BooleanClause clause : clauses) {
            bqb.add(clause);
          }
          return bqb.build();
        });
  }

  public static SpanQuery getSpanQuery(CharSequence string,
                                       String field,
                                       Analyzer analyzer) {
    final List<String> tokens = tokenize(string, field, analyzer);
    final SpanQuery[] clauses = new SpanQuery[tokens.size()];
    for (int i = 0; i < clauses.length; i++) {
      SpanTermQuery term = new SpanTermQuery(new Term(field, tokens.get(i)));
      clauses[i] = term;
    }
    return new SpanNearQuery(clauses, 3, false);
  }

  private static SpanQuery[] getSpanQueryClauses(CharSequence string,
                                                 String field,
                                                 Analyzer analyzer,
                                                 double weight) {
    final List<String> tokens = tokenize(string, field, analyzer);
    final SpanQuery[] clauses = new SpanQuery[ tokens.size() ];
    for (int i = 0; i < clauses.length; i++) {
      SpanTermQuery term = new SpanTermQuery(new Term(field, tokens.get(i)));
      clauses[i] = new SpanBoostQuery(term, (float) weight);
    }
    log.trace("Creating SpanQuery of {}", Arrays.toString(clauses));
    return clauses;
  }

  private static SpanQuery[] getUnweightedSpanQueryClauses(CharSequence string,
                                                           String field,
                                                           Analyzer analyzer) {
    final List<String> tokens = tokenize(string, field, analyzer);
    final SpanQuery[] clauses = new SpanQuery[ tokens.size() ];
    for (int i = 0; i < clauses.length; i++) {
      clauses[i] = new SpanTermQuery(new Term(field, tokens.get(i)));
    }
    log.trace("Creating SpanQuery of {}", Arrays.toString(clauses));
    return clauses;
  }

  public static Query getSpanQuery(CharSequence string,
                                   String field,
                                   Analyzer analyzer,
                                   double weight) {
    final SpanQuery[] clauses = getSpanQueryClauses(string, field, analyzer, weight);
    if (clauses.length == 0) {
      return new BooleanQuery.Builder().build();
    } else if (clauses.length == 1) {
      return clauses[0];
    } else {
      return new SpanNearQuery(clauses, clauses.length / 2, false);
    }
  }


  private static Optional<Query> tryBuildSpanNearQuery(SpanQuery[] clauses,
                                                       int slop, boolean inOrder) {
    if (clauses.length > 1) {
      return Optional.of(new SpanNearQuery(clauses, slop, inOrder));
    } else if (clauses.length == 1) {
      return Optional.of(clauses[0]);
    } else return Optional.empty();
  }

  public static Optional<Query> getUnweightedSubQuery(CharSequence string,
                                                      String field,
                                                      Analyzer analyzer,
                                                      int slop, boolean inOrder) {
    return tryBuildSpanNearQuery(getUnweightedSpanQueryClauses(string, field, analyzer),
        slop,
        inOrder);

  }

  public static Optional<Query> getSubQuery(CharSequence string,
                                            String field,
                                            Analyzer analyzer,
                                            double weight, int slop, boolean inOrder) {
    return tryBuildSpanNearQuery(getSpanQueryClauses(string, field, analyzer, weight),
        slop,
        inOrder);
  }

  public static Optional<Query> getSubQuery(CharSequence string,
                                            String field,
                                            Analyzer analyzer,
                                            double weight) {
    return getSubQuery(string, field, analyzer, weight, 3, false);
  }

  public static LoadingCache<Triple<Weighted<? extends CharSequence>,
                             String,
                             Analyzer>, Optional<Query>> subqueryCache =
      CacheBuilder.newBuilder()
      .maximumSize(1000)
      .build(CacheLoader.from(w -> {
        assert w != null;
        return getSubQuery(w.getLeft().value, w.getMiddle(), w.getRight(), w.getLeft().weight);
      }));

  public static <T extends CharSequence, K extends Weighted<T>> BooleanQuery getBooleanQuery(
      Iterable<K> it, String field, Analyzer analyzer, BooleanClause.Occur occur) {
    final BooleanQuery.Builder query = new BooleanQuery.Builder();
    for (Weighted<? extends CharSequence> term : it) {
      final Optional<Query> subQuery = subqueryCache.getUnchecked(Triple.of(term, field, analyzer));
      subQuery.ifPresent(q -> query.add(q, occur));
    }
    log.trace("Created BooleanQuery {}", query);
    return query.build();
  }

  public static int getNumberOfTerms(IndexSearcher searcher, org.apache.lucene.search.Query query) {
    final Set<Term> terms = Sets.newHashSet();
    try {
      query.rewrite(searcher.getIndexReader())
          .createWeight(searcher, false).extractTerms(terms);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return terms.size();
  }

  public static <T extends BaseDocument> Document<T> asScribe(
      org.apache.lucene.document.Document document,
      String textField,
      String idField) {
    final List<IndexableField> fields = document.getFields();
    Multimap<String, String> map = HashMultimap.create();
    List<String> text;
    String value;
    for (IndexableField field : fields) {
      value = field.stringValue();
      if (value != null) {
        map.put(field.name(), value);
      }
    }

    final Joiner joiner = Joiner.on('\n');
    final Document<T> doc = Document.fromString(joiner.join(map.get(textField)).trim());
    map.removeAll(textField);

    doc.set(BaseDocument.id, joiner.join(map.get(idField)).trim());
    map.removeAll(idField);

    for (String key : map.keySet()) {
      doc.features.put(key, joiner.join(map.get(key)).trim());
    }
    return doc;
  }


  public static LocalDate getLocalDateField(org.apache.lucene.document.Document luceneDocument,
      String field) {
    final IndexableField dateField = luceneDocument.getField(field);
    if (dateField == null) {
      return null;
    } else {
      return LocalDate.ofEpochDay(dateField.numericValue().longValue());
    }
  }

  public static List<String> getListField(org.apache.lucene.document.Document luceneDocument,
      String field) {
    return Arrays.stream(luceneDocument.getFields(field))
        .map(IndexableField::stringValue)
        .collect(Collectors.toList());
  }
}
