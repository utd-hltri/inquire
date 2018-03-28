/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.utdallas.hltri.inquire.eval;

import com.google.common.base.CharMatcher;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ForwardingMap;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import edu.utdallas.hltri.inquire.SearchResult;
import edu.utdallas.hltri.inquire.SimpleSearchResult;

/**
 *
 * @author travis
 */
public class TrecRunReader extends ForwardingMap<String, List<SearchResult<String>>> {
  final private Path path;

  @SuppressWarnings("unused")
  public TrecRunReader(String path) {
    this(Paths.get(path));
  }

  @SuppressWarnings("unused")
  public TrecRunReader(File file) {
    this(file.toPath());
  }

  private Map<String, List<SearchResult<String>>> results;

  private String runtag = null;

  public TrecRunReader(Path path) {
    Preconditions.checkArgument(Files.notExists(path) || Files.isWritable(path),
        "path is not writable");
    this.path = path;
    this.results = parseResults();
  }

  private Map<String, List<SearchResult<String>>> parseResults() {
    final Splitter splitter = Splitter.on(CharMatcher.whitespace()).omitEmptyStrings().trimResults().limit(6);
    final Map<String, List<SearchResult<String>>> run = new LinkedHashMap<>();
    try (BufferedReader reader = Files.newBufferedReader(path)) {
      for (String line; ((line = reader.readLine()) != null); ) {
        final List<String> fields = splitter.splitToList(line);
        assert fields.get(1).equals("Q0");
        final String topicId = fields.get(0);
        final List<SearchResult<String>> results = run.computeIfAbsent(topicId, k -> new ArrayList<>());
        final String docId = fields.get(2);
        final int rank = Integer.parseInt(fields.get(3));
        final double score = Double.parseDouble(fields.get(4));
        final String runtag = fields.get(5);
        if (this.runtag == null) {
          this.runtag = runtag;
        } else {
          assert this.runtag.equals(runtag);
        }
        results.add(new SimpleSearchResult<>(rank, score, docId));
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return run;
  }

  @Override
  protected Map<String, List<SearchResult<String>>> delegate() {
    return results;
  }

  public String getRuntag() {
    return runtag;
  }
}
