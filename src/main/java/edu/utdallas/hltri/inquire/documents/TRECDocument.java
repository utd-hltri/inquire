package edu.utdallas.hltri.inquire.documents;

import edu.utdallas.hltri.scribe.text.BaseDocument;
import edu.utdallas.hltri.scribe.text.DocumentAttribute;
import edu.utdallas.hltri.scribe.text.documents.HasTitle;

/**
 * Created by travis on 11/6/15.
 */
public class TRECDocument extends BaseDocument {
  public static DocumentAttribute<TRECDocument, String> sgmlFeature(String feature) {
    return DocumentAttribute.typed(feature, String.class);
  }
}
