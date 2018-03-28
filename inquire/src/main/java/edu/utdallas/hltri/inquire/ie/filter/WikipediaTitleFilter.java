package edu.utdallas.hltri.inquire.ie.filter;

import com.google.common.base.CharMatcher;
import com.google.common.base.Splitter;
import com.google.common.base.Strings;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import edu.utdallas.hltri.conf.Config;
import edu.utdallas.hltri.func.CloseablePredicate;
import edu.utdallas.hltri.io.ExternalBinarySearch;
import edu.utdallas.hltri.io.StusMagicLargeFileReader;
import edu.utdallas.hltri.logging.Logger;

import static com.google.common.base.CharMatcher.JAVA_LETTER_OR_DIGIT;

/**
 * Created by travis on 7/23/14.
 */
public class WikipediaTitleFilter implements CloseablePredicate<CharSequence> {

  private static final Logger log = Logger.get(WikipediaTitleFilter.class);

  private final CharMatcher matcher  = JAVA_LETTER_OR_DIGIT.or(CharMatcher.is('/')).negate();
  private final Splitter    splitter = Splitter.on(matcher).omitEmptyStrings();

  private final Config conf = Config.load("inquire.wikipedia.titles");


  private final StusMagicLargeFileReader wikiTitles;

  public WikipediaTitleFilter() {
    try {
      final File wikipediaFile = conf.getFile("path");
      log.debug("Loading Wikipedia data from {}", wikipediaFile);
      wikiTitles = new StusMagicLargeFileReader(wikipediaFile);
    } catch (IOException ex) {
      throw new RuntimeException(ex);
    }
  }

  @Override public void close() {
    try {
      wikiTitles.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override public boolean test(CharSequence input) {
    final String string = input.toString();
    assert !Strings.isNullOrEmpty(string) : "string was empty";
    final Iterator<String> it = splitter.split(string).iterator();
    if (!it.hasNext()) {
      log.debug("|{}| had no valid characters", input);
      return false;
    }
    String prefix = it.next();
    if (prefix.length() == 0) {
      prefix = matcher.removeFrom(string);
    }
    final String formatted = matcher.replaceFrom(input, ' ');
    synchronized (wikiTitles) {
      for (final String word : ExternalBinarySearch.binarySearch(wikiTitles, prefix)) {
        if (matcher.replaceFrom(word, ' ').equalsIgnoreCase(formatted)) {
          log.debug("|{}| matched Wikipedia title", string);
          return true;
        }
      }
    }
    log.debug("|{}| failed Wikipedia filter", string);
    return false;
  }
}
