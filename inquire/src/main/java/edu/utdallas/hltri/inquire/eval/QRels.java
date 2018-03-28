
package edu.utdallas.hltri.inquire.eval;

import com.google.common.base.CharMatcher;
import com.google.common.base.Splitter;
import com.google.common.base.Supplier;
import com.google.common.collect.Iterables;
import com.google.common.collect.Table;
import com.google.common.collect.Tables;
import edu.utdallas.hltri.logging.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 *
 * @author travis
 */
public class QRels {

  transient private static final Logger log = Logger.get(QRels.class);

  /**
   * Enumeration of possible relevance values:
   * <dl>
   *  <dt>NONRELEVANT
   *  <dd>The visit was judged as not relevant for this question
   *
   *  <dt>PARTIAL
   *  <dd>The visit was judged as partially relevant for this question
   *      (this is considered relevant)
   *
   *  <dt>RELEVANT
   *  <dd>The visit was judged as relevant for this question
   *
   *  <dt>UNKNOWN
   *  <dd>The visit was not judged for this question
   * </dl>
   */
  public enum Relevance {
    NONRELEVANT,
    PARTIAL,
    RELEVANT,
    UNKNOWN;

    public int toInt() {
      switch (this) {
        case PARTIAL:
          return 1;
        case RELEVANT:
          return 2;
        case NONRELEVANT:
        case UNKNOWN:
        default:
          return 0;
      }
    }

    public int toCollapsedInt() {
      switch(this) {
        case PARTIAL:
        case RELEVANT:
          return 1;
        case NONRELEVANT:
        case UNKNOWN:
        default:
          return 0;
      }
    }

    public boolean toBoolean() {
      return toCollapsedInt() == 1;
    }
  }

  /*
   * Create a table that remembers the order of both the questions and
   * the visits when reading the question relevance files.
   *
   * This is necessary because MAP requires results being in order,
   * and we'd also like to be able to iterate questions in order.
   *
   * Because guava has no factory for a linked hash table, we have to do
   * it ourselves.
   */
  private final Table<String, String, Relevance> qrels = Tables.newCustomTable(
          new LinkedHashMap<String, Map<String, Relevance>>(),
          new Supplier<Map<String, Relevance>>() {
    @Override public LinkedHashMap<String, Relevance> get() {
      return new LinkedHashMap<>();
    }
  });
  private final Path path;

  public static QRels fromFile(final String string) {
    return new QRels(Paths.get(string));
  }

  public static QRels fromFile(final Path path) {
    return new QRels(path);
  }

  /**
   * Creates a QRels object from the question relevance file at the given Path
   * @param path Path containing the question relevances
   */
  protected QRels(Path path) {
    this.path = path;
    parse();
  }

  private void parse() {
    final Relevance[] values = Relevance.values();
    final Splitter splitter = Splitter.on(CharMatcher.WHITESPACE).limit(5).omitEmptyStrings();
    try (BufferedReader reader = Files.newBufferedReader(path, Charset.defaultCharset())) {
      for (String line; (line = reader.readLine()) != null; ) {
        String[] fields = Iterables.toArray(splitter.split(line), String.class);
        try {
          qrels.put(fields[0], fields[2], values[Integer.valueOf(fields[3])]);
        } catch (ArrayIndexOutOfBoundsException ex) {
          qrels.put(fields[0], fields[2], Relevance.UNKNOWN);
        }
      }
    } catch (IOException ex) {
      throw new RuntimeException(ex);
    }
    log.debug("Parsed {} qrels.", qrels.size());
  }

  /**
   * Returns the relevance of the indicated visit for the given question
   * @param topic_id Question id
   * @param doc_id Visit id
   * @return Relevance of the desired visit for the given question
   */
  public Relevance getRelevance(String topic_id, String doc_id) {
    if (qrels.contains(topic_id, doc_id)) {
      return qrels.get(topic_id, doc_id);
    } else {
      return Relevance.UNKNOWN;
    }
  }

  /**
   * Returns the qrel information for the given question
   * @param topic_id Topic id
   * @return Map of qrel information
   */
  public Map<String, Relevance> getJudgements(String topic_id) {
    return qrels.row(topic_id);
  }

  public Set<String> getTopicsWithDocument(String documentId) {
    return qrels.column(documentId).keySet();
  }

  /**
   * Returns the set of questions in this QRels object
   * @return Set of question ids
   */
  public Set<String> getTopics() {
    return qrels.rowKeySet();
  }

  public Set<String> getJudgedDocuments() {
    final Set<String> visits = new HashSet<>();
    for (final String question : getTopics()) {
      visits.addAll(getJudgements(question).keySet());
    }
    return Collections.unmodifiableSet(visits);
  }

  public Path getPath() {
    return path;
  }
}
