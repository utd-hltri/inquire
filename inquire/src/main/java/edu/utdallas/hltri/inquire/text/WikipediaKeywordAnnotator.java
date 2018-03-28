package edu.utdallas.hltri.inquire.text;

import com.google.common.base.CharMatcher;
import com.google.common.base.Splitter;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import edu.utdallas.hltri.conf.Config;
import edu.utdallas.hltri.func.CloseablePredicate;
import edu.utdallas.hltri.io.ExternalBinarySearch;
import edu.utdallas.hltri.io.StusMagicLargeFileReader;
import edu.utdallas.hltri.logging.Logger;
import edu.utdallas.hltri.scribe.annotators.Annotator;
import edu.utdallas.hltri.scribe.text.BaseDocument;
import edu.utdallas.hltri.scribe.text.Document;
import edu.utdallas.hltri.scribe.text.DuplicateAnnotationException;
import edu.utdallas.hltri.scribe.text.Text;
import edu.utdallas.hltri.scribe.text.annotation.Token;
import edu.utdallas.hltri.util.LongestSequenceIterable;

import static com.google.common.base.CharMatcher.JAVA_LETTER_OR_DIGIT;

/**
 *
 * @author travis
*/
public class WikipediaKeywordAnnotator<D extends BaseDocument> implements Annotator<D> {
  private static final Logger log = Logger.get(WikipediaKeywordAnnotator.class);

  private final StusMagicLargeFileReader wikiTitles;

  private final CharMatcher matcher  = JAVA_LETTER_OR_DIGIT.or(CharMatcher.is('/')).negate();
  private final Splitter    splitter = Splitter.on(matcher).omitEmptyStrings();

  private static final Config conf = Config.load("inquire.wikipedia");

  private final CloseablePredicate<CharSequence> filter;

  public WikipediaKeywordAnnotator(CloseablePredicate<CharSequence> filter) {
    try {
      final File wikipediaFile = conf.getFile("titles.path");
      log.debug("Loading Wikipedia data from {}", wikipediaFile);
      wikiTitles = new StusMagicLargeFileReader(wikipediaFile);
      this.filter = filter;
    } catch (IOException ex) {
      throw new RuntimeException(ex);
    }
  }

  private boolean clear = false;

  @SuppressWarnings("unused")
  public WikipediaKeywordAnnotator<D> clear() {
    this.clear = true;
    return this;
  }


  @Override public <B extends D> void annotate(Document<B> document) {
    if (!document.get(Keyword.TYPE).isEmpty()) {
      if (clear) {
        document.clear(Keyword.TYPE);
      } else {
        log.warn("Skipping {} because it already has keywords", document.get(BaseDocument.id));
      }
    }
    if (document.get("genia", Token.TYPE).isEmpty()) {
      log.error("No tokens found on {}", document.get(BaseDocument.id));
      throw new IllegalArgumentException();
    }
    annotateKeywords(document);
  }

  @SuppressWarnings("unused")
  public static <B extends Text> void condense(Iterable<B> inner) {
    for (final Iterator<B> it = inner.iterator(); it.hasNext(); ) {
      final B itKeyword = it.next();
      final String itString = itKeyword.asString().toLowerCase();

      boolean contains = false;
      for (final B jtKeyword : inner) {
        final String jtString = jtKeyword.asString().toLowerCase();

        // Compare against all other elements
        if (!jtString.equals(itString) && jtString.contains(itString)) {
          contains = true;
          break;
        }
      }

      if (contains) {
        it.remove();
      }
    }
  }

  @SuppressWarnings("WeakerAccess")
  public void annotateKeywords(Document<?> doc) {
    final List<Token> tokens = doc.get("genia", Token.TYPE);

    String keyword;
    long start, end;
    for (final List<Token> seq : LongestSequenceIterable.of(tokens)) {
      start = seq.get(0).get(Token.StartOffset);
      end = seq.get(seq.size() - 1).get(Token.EndOffset);
      keyword = doc.subString(start, end).toLowerCase();

      if (JAVA_LETTER_OR_DIGIT.matchesNoneOf(seq.get(0).asString()) ||
          JAVA_LETTER_OR_DIGIT.matchesNoneOf(seq.get(seq.size() - 1).asString()) ||
          !filter.test(keyword)) {
        continue;
      }

      if (isWikipediaTitle(keyword)) {
        try {
          Keyword.TYPE.create(doc, "wiki", start, end);
        } catch (DuplicateAnnotationException e) {
          // ignore;
        }
      }
    }
  }

  private boolean isWikipediaTitle(CharSequence keyword) {
    String prefix = splitter.split(keyword).iterator().next();
    if (prefix.length() == 0) { prefix = matcher.removeFrom(keyword); }
    final String formatted = matcher.replaceFrom(keyword, ' ');
    synchronized(wikiTitles) {
      for (final String word : ExternalBinarySearch.binarySearch(wikiTitles, prefix)) {
        if (matcher.replaceFrom(word, ' ').equalsIgnoreCase(formatted)) {
          return true;
        }
      }
    }
    return false;
  }
}


