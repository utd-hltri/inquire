package edu.utdallas.hltri.inquire.text.annotators;

import edu.utdallas.hltri.inquire.text.Keyword;
import edu.utdallas.hltri.logging.Logger;
import edu.utdallas.hltri.scribe.annotators.Annotator;
import edu.utdallas.hltri.scribe.text.BaseDocument;
import edu.utdallas.hltri.scribe.text.Document;
import edu.utdallas.hltri.scribe.text.DuplicateAnnotationException;
import edu.utdallas.hltri.scribe.text.annotation.Annotation;
import edu.utdallas.hltri.scribe.text.annotation.Sentence;
import edu.utdallas.hltri.scribe.text.annotation.Token;
import edu.utdallas.hltri.util.LongestSequenceIterable;

import java.util.List;
import java.util.function.Predicate;

/**
 * Created by travis on 7/23/14.
 */
public class KeywordAnnotator <D extends BaseDocument> implements Annotator<D> {
  private static final Logger log = Logger.get(KeywordAnnotator.class);

  private final String                 tokenAnnotationSet;
  private final String                 sentenceAnnotationSet;
  private final String                 keywordAnnotationSet;
  private final Predicate<List<Token>>  tokenFilter;
  private final Predicate<CharSequence> textFilter;

  private boolean clear = false;

  public KeywordAnnotator(final String tokenAnnotationSet,
                          final Predicate<List<Token>> tokenFilter,
                          final Predicate<CharSequence> textFilter,
                          final String keywordAnnotationSet) {
    this(tokenAnnotationSet, tokenAnnotationSet, tokenFilter, textFilter, keywordAnnotationSet);
  }

  public KeywordAnnotator(final String tokenAnnotationSet,
                          final String sentenceAnnotationSet,
                          final Predicate<List<Token>> tokenFilter,
                          final Predicate<CharSequence> textFilter,
                          final String keywordAnnotationSet) {
    this.tokenAnnotationSet = tokenAnnotationSet;
    this.sentenceAnnotationSet = sentenceAnnotationSet;
    this.keywordAnnotationSet = keywordAnnotationSet;
    this.tokenFilter = tokenFilter;
    this.textFilter = textFilter;
  }

  public KeywordAnnotator clear() {
    this.clear = true;
    return this;
  }

  @Override public <B extends D> void annotate(Document<B> document) {
    if (this.clear) {
      document.clear(keywordAnnotationSet, Keyword.TYPE);
    }
    log.debug("Annotating {}", document);
    long start, end;
    String string;
    for (final Sentence sentence : document.get(sentenceAnnotationSet, Sentence.TYPE)) {
      log.debug("Annotating sentence {}", sentence);

      for (final List<Token> seq : LongestSequenceIterable.of(sentence.getContained(tokenAnnotationSet, Token.TYPE))) {
        start = seq.get(0).get(Annotation.StartOffset);
        end = seq.get(seq.size() - 1).get(Annotation.EndOffset);
        string = document.subString(start, end).toLowerCase();

        if (!tokenFilter.test(seq)) {
          log.trace("|{}| failed token filter {}", string, tokenFilter);
          continue;
        }

        if (!textFilter.test(string)) {
          log.trace("|{}| failed text filter {}", string, textFilter);
          continue;
        }

        try {
          Keyword.TYPE.create(document, keywordAnnotationSet, start, end);
        } catch (DuplicateAnnotationException e) {
          // do nothing
        }
      }
    }
  }
}
