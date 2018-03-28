package edu.utdallas.hltri.inquire.ie.filter;

import com.google.common.base.Joiner;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

import javax.annotation.Nullable;

import edu.utdallas.hltri.conf.Config;
import edu.utdallas.hltri.func.CloseablePredicate;
import edu.utdallas.hltri.logging.Logger;

/**
 * Created by travis on 7/11/14.
 */
public class IgnoreWordTextFilter implements CloseablePredicate<List<? extends CharSequence>> {
  private static final Logger log = Logger.get();

  private final Collection<String> ignoreSet = new HashSet<>();

  final Config conf            = Config.load("inquire.filter");
  final Path   ignoreSetSource = conf.getPath("ignore-path");

  public IgnoreWordTextFilter() {
    try {
      ignoreSet.addAll(Files.readAllLines(ignoreSetSource, Charset.defaultCharset()));
    } catch (IOException ex) {
      throw new RuntimeException("Failed to open ignore word file at " + ignoreSetSource, ex);
    }
  }

  @Override public boolean test(@Nullable List<? extends CharSequence> tokens) {
    if (tokens == null) return false;
    String string;
    for (final CharSequence token : tokens) {
      string = token.toString().toLowerCase();
      for (final String ignore : ignoreSet) {
        if (ignore.equals(string)) {
          log.trace("|{}| from |{}| == |{}|", string, Joiner.on(' ').join(tokens), ignore);
          return false;
        }
      }
    }
    return true;
  }

  @Override public void close() {
    ignoreSet.clear();
  }
}
