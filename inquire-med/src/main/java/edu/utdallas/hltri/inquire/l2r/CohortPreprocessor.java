package edu.utdallas.hltri.inquire.l2r;

import java.nio.file.Path;

import edu.utdallas.hltri.inquire.lucene.LuceneStopwords;
import edu.utdallas.hltri.inquire.text.WikipediaKeywordAnnotator;
import edu.utdallas.hltri.scribe.annotators.GeniaAnnotator;
import edu.utdallas.hltri.scribe.annotators.LingScopeNegationSpanAnnotator;
import edu.utdallas.hltri.scribe.annotators.OpenNLPSentenceAnnotator;
import edu.utdallas.hltri.scribe.io.JsonCorpus;
import edu.utdallas.hltri.scribe.text.BaseDocument;
import edu.utdallas.hltri.scribe.text.Document;
import edu.utdallas.hltri.scribe.text.annotation.Sentence;
import edu.utdallas.hltri.scribe.text.annotation.Token;

/**
 * Created by travis on 11/4/16.
 */
public class CohortPreprocessor<K extends BaseDocument> {
  private final JsonCorpus<K> corpus;

  private final OpenNLPSentenceAnnotator<K> ssplit = new OpenNLPSentenceAnnotator<>();

  private final GeniaAnnotator<K> tokenizer = new GeniaAnnotator.Builder<K>()
      .annotateTokens()
      .withSentences(d -> d.get(OpenNLPSentenceAnnotator.ANNOTATION_SET_NAME, Sentence.TYPE))
      .build();

  private final WikipediaKeywordAnnotator<K> wiki =
      new WikipediaKeywordAnnotator<>(LuceneStopwords.LUCENE::test);

  private final LingScopeNegationSpanAnnotator<K> lingscope =
      new LingScopeNegationSpanAnnotator<>(
          d -> d.get(OpenNLPSentenceAnnotator.ANNOTATION_SET_NAME, Sentence.TYPE),
          s -> s.getContained(GeniaAnnotator.ANNOTATION_SET_NAME, Token.TYPE));

  public CohortPreprocessor(Path path) {
    this.corpus = JsonCorpus.<K>at(path).build();
  }

  public void preprocess(Document<K> doc) {
    doc.setCorpus(corpus);
    ssplit.annotate(doc);
    tokenizer.annotate(doc);
    wiki.annotate(doc);
    lingscope.annotate(doc);
    doc.sync();
  }

  public JsonCorpus<K> getCorpus() {
    return corpus;
  }
}
