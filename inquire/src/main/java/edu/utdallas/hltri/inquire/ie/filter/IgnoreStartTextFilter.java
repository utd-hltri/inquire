package edu.utdallas.hltri.inquire.ie.filter;

import com.google.common.base.Charsets;
import com.google.common.base.Joiner;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

import edu.utdallas.hltri.conf.Config;
import edu.utdallas.hltri.func.CloseablePredicate;
import edu.utdallas.hltri.logging.Logger;
import edu.utdallas.hltri.scribe.text.annotation.Token;

/**
 * Created by travis on 7/11/14.
 */
public class IgnoreStartTextFilter implements CloseablePredicate<List<Token>> {
  private final static Logger             log       = Logger.get();
  private final        Collection<String> ignoreSet = new HashSet<>();

  final Config conf            = Config.load("inquire.filter");
  final Path   ignoreSetSource = Paths.get(conf.getString("ignore-start-path"));

  public IgnoreStartTextFilter() {
    try {
      ignoreSet.addAll(Files.readAllLines(ignoreSetSource, Charsets.UTF_8));
    } catch (IOException ex) {
      throw new RuntimeException("Failed to open ignore word file at " + ignoreSetSource, ex);
    }
  }

  @Override public boolean test(List<Token> tokens) {
    if (tokens == null) return false;
    final String string = tokens.get(0).asString().toLowerCase();
    for (final String ignore : ignoreSet) {
      if (string.equals(ignore)) {
        log.trace("|{}| from |{}| == |{}|", string, Joiner.on(' ').join(tokens), ignore);
        return false;
      }
    }
    return true;
  }

  @Override public void close() {
    ignoreSet.clear();
  }
}
