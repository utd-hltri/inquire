package edu.utdallas.hltri.inquire.documents;

import edu.utdallas.hltri.scribe.text.BaseDocument;
import edu.utdallas.hltri.scribe.text.DocumentAttribute;
import edu.utdallas.hltri.scribe.text.documents.HasTitle;

/**
 * Created by travis on 10/2/15.
 */
public class NewsArticle extends BaseDocument implements HasTitle {
  public static final DocumentAttribute<BaseDocument, String>
      dateline = DocumentAttribute.typed("dateline", String.class);

  public static final DocumentAttribute<BaseDocument, String>
      byline = DocumentAttribute.typed("byline", String.class);
}
