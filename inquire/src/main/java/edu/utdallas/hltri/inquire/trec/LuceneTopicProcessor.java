//package edu.utdallas.hltri.inquire.trec;
//
//import org.apache.lucene.search.BooleanClause;
//import org.apache.lucene.search.BooleanQuery;
//import org.apache.lucene.search.Query;
//
//import java.util.ArrayList;
//import java.util.Collection;
//import java.util.List;
//
//import edu.utdallas.hltri.inquire.Aspect;
//import edu.utdallas.hltri.inquire.text.Keyword;
//
///**
// * Created by travis on 11/1/16.
// */
//public class LuceneTopicProcessor {
//
//  public Collection<Query> coverToLucene(Topic topic) {
//    Collection<org.apache.lucene.search.Query> queries = new ArrayList<>();
//
//    for (Aspect aspect : topic.getAspects()) {
//      if (aspect.isNegated() || !aspect.isRequired()) {
//        continue;
//      }
//
//      BooleanQuery q;
//
//      if (aspect.isHasDisjunction()) {
//        q = new BooleanQuery();
//        List<Aspect> disjunctionGroup = aspect.getDisjunctionGroup();
//        if (disjunctionGroup.indexOf(aspect) == 0) {
//          for (Keyword disjunctionMember : disjunctionGroup) {
//            q.add(disjunctionMember.asLuceneQuery(), BooleanClause.Occur.SHOULD);
//          }
//        } else {
//          continue;
//        }
//      } else {
//        q = keyword.asLuceneQuery();
//      }
//
//      queries.add(q);
//    }
//
//    return queries;
//  }
//
//
//  public BooleanQuery asLuceneQuery() {
//    return asLuceneSpanQuery();
//  }
//
//  public BooleanQuery asLuceneSpanQuery() {
//    return toLuceneQuery(new LuceneSpanPhraseConverter());
//  }
//
//  public BooleanQuery asLucenePhraseQuery() {
//    return toLuceneQuery(new LucenePhrasePhraseConverter());
//  }
//
//
//  private BooleanQuery toLuceneQuery(LucenePhraseConverter<?> converter) {
//    BooleanQuery bq = new BooleanQuery();
//
//    for (Keyword keyword : keywords) {
//      if (keyword.isNegation()) {
//        continue;
//      }
//
//      org.apache.lucene.search.Query q = keyword.toLuceneQuery(converter);
//
//      if (keyword.isDisjunction()) {
//        q = new BooleanQuery();
//        List<Keyword> disjunctionGroup = keyword.getDisjunctionGroup();
//        if (disjunctionGroup.indexOf(keyword) == 0) {
//          for (Keyword disjunctionMember : disjunctionGroup) {
//            ((BooleanQuery) q).add(disjunctionMember.asLuceneQuery(), BooleanClause.Occur.SHOULD);
//          }
//        } else {
//          continue;
//        }
//      }
//
//      if (keyword.isRequired()) {
//        bq.add(q, BooleanClause.Occur.SHOULD);
//      } else {
//        bq.add(q, BooleanClause.Occur.SHOULD);
//      }
//    }
//
//    return bq;
//  }
//}
