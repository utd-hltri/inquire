//package edu.utdallas.hltri.inquire.trec;
//
//import com.google.common.collect.Sets;
//import com.google.common.primitives.Doubles;
//
//import org.apache.lucene.search.BooleanClause.Occur;
//import org.apache.lucene.search.BooleanQuery;
//
//import java.io.Serializable;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.Collection;
//import java.util.Iterator;
//import java.util.List;
//import java.util.Objects;
//import java.util.Set;
//import java.util.TreeSet;
//
//import edu.utdallas.hlt.trecmed.retrieval.LuceneEMRSearcher;
//import edu.utdallas.hlt.trecmed.retrieval.LucenePhraseConverter;
//import edu.utdallas.hlt.trecmed.retrieval.LucenePhrasePhraseConverter;
//import edu.utdallas.hlt.trecmed.retrieval.LuceneSpanPhraseConverter;
//import edu.utdallas.hltri.inquire.Aspect;
//import edu.utdallas.hltri.inquire.expansion.WeightedExpansion;
//import edu.utdallas.hltri.inquire.ie.AgeExtractor.AgeRange;
//import edu.utdallas.hltri.inquire.ie.GenderExtractor.Gender;
//import edu.utdallas.hltri.inquire.text.Keyword;
//import edu.utdallas.hltri.logging.Logger;
//import edu.utdallas.hltri.scribe.text.Text;
//import edu.utdallas.hltri.struct.Weighted;
//
///**
// *
// * @author travis
// */
//public class Topic extends Text implements Iterable<Aspect>, Serializable {
//  private static final Logger log = Logger.get(Topic.class);
//
//
//  private static final long serialVersionUID = 1L;
//
//  private final     String                     id;
//  private final     String                     rawText;
//
//  private final Set<Aspect> aspects = new TreeSet<>();
//
//  public Topic(String id, String rawText) {
//    this.id = id;
//    this.rawText = rawText;
//  }
//
//  public String getId() {
//    return id;
//  }
//
//  @Override public String describe () {
//    return "Topic #" + id + ": '" + rawText + "'";
//  }
//
//  @Override public Iterator<Aspect> iterator() {
//    return aspects.iterator();
//  }
//
//  @Override public String asString() {
//    return this.rawText;
//  }
//
//  public Set<Aspect> getAspects() {
//    return aspects;
//  }
//
//
//
//  @Override public boolean equals(Object obj) {
//    if (obj == null) {
//      return false;
//    }
//    if (getClass() != obj.getClass()) {
//      return false;
//    }
//    final Topic other = (Topic) obj;
//    if (!Objects.equals(this.id, other.id)) {
//      return false;
//    }
//    if (!Objects.equals(this.rawText, other.rawText)) {
//      return false;
//    }
//    return true;
//  }
//
//  @Override public int hashCode() {
//    int hash = 5;
//    hash = 67 * hash + Objects.hashCode(this.id);
//    hash = 67 * hash + Objects.hashCode(this.rawText);
//    return hash;
//  }
//}
