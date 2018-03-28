package edu.utdallas.hltri.inquire.lucene;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.spans.SpanNearQuery;
import org.apache.lucene.search.spans.SpanQuery;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Spliterators;
import java.util.stream.Collector;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class LuceneStreamUtils {

  public static Iterator<String> getTokenIterator(final CharSequence text,
      final String field,
      final Analyzer analyzer) {
    final TokenStream stream = analyzer.tokenStream(field, text.toString());
    try {
      stream.reset();
    } catch (IOException ex) {
      throw new RuntimeException("Lucene said this should never be thrown!", ex);
    }
    return new Iterator<String>() {
      @Override
      public boolean hasNext() {
        try {
          boolean hastNext = stream.incrementToken();
          if (!hastNext) {
            stream.close();
          }
          return hastNext;
        } catch (IOException ex) {
          throw new RuntimeException("Lucene said this should never be thrown!", ex);
        }
      }

      @Override
      public String next() {
        return stream.getAttribute(CharTermAttribute.class).toString();
      }
    };
  }


  public static Stream<String> streamTokens(
      final CharSequence text,
      final String field,
      final Analyzer analyzer) {
    return StreamSupport.stream(Spliterators.spliteratorUnknownSize(
        getTokenIterator(text, field, analyzer), 0), false);
  }

  public static Stream<Term> streamTerms(
      final CharSequence text,
      final String field,
      final Analyzer analyzer) {
    return streamTokens(text, field, analyzer).map(token -> new Term(field, token));
  }

  public static Stream<TermQuery> streamTermsAsQueries(
      final CharSequence text,
      final String field,
      final Analyzer analyzer) {
    return streamTokens(text, field, analyzer).map(token -> new TermQuery(new Term(field, token)));
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

  public static <T extends SpanQuery> Collector<T, ArrayList<SpanQuery>, SpanNearQuery> toSpanNearQuery(
      final int slop,
      final boolean inOrder) {
    return Collector.of(
        ArrayList::new,
        ArrayList::add,
        (clauses1, clauses2) -> {
          clauses1.addAll(clauses2);
          return clauses1;
        },
        clauses -> new SpanNearQuery(clauses.toArray(new SpanQuery[ clauses.size() ]),
              slop,
              inOrder)
    );
  }
}
