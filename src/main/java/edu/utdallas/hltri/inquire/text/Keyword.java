package edu.utdallas.hltri.inquire.text;

import edu.utdallas.hltri.scribe.text.annotation.AbstractAnnotation;
import edu.utdallas.hltri.scribe.text.annotation.AbstractAnnotationType;
import edu.utdallas.hltri.scribe.text.annotation.AnnotationType;
import edu.utdallas.hltri.scribe.text.Document;
import gate.Annotation;

/**
 * Created by travis on 7/15/14.
 */
public class Keyword extends AbstractAnnotation<Keyword> {
  public Keyword(Document document, Annotation gateAnnotation) {
    super(document, gateAnnotation);
  }

  public static AnnotationType<Keyword> TYPE = new AbstractAnnotationType<Keyword>("Keyword") {
    @Override public Keyword wrap(Document parent, Annotation annotation) {
      return new Keyword(parent, annotation);
    }
  };
}

