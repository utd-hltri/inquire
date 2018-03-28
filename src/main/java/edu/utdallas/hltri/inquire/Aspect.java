package edu.utdallas.hltri.inquire;

import com.google.common.collect.Iterables;

import java.util.Iterator;
import java.util.Set;

import edu.utdallas.hltri.inquire.text.Keyword;
import edu.utdallas.hltri.logging.Logger;
import edu.utdallas.hltri.scribe.text.Text;

/**
 * Created by travis on 11/1/16.
 */
public class Aspect extends Text {
  private static final Logger log = Logger.get(Aspect.class);

  final String rawString;

  boolean negated = false;
  boolean required = true;
  boolean hasDisjunction = false;

  public Aspect(String rawString) {
    this.rawString = rawString;
  }

  public boolean isNegated() {
    return negated;
  }

  public void setNegated(boolean negated) {
    this.negated = negated;
  }

  public boolean isRequired() {
    return required;
  }

  public void setRequired(boolean required) {
    this.required = required;
  }

  public boolean isHasDisjunction() {
    return hasDisjunction;
  }

  public void setHasDisjunction(boolean hasDisjunction) {
    this.hasDisjunction = hasDisjunction;
  }

  @Override
  public String asString() {
    return rawString;
  }

  public static void filterByWords(Iterable<Aspect> aspects, Set<String> filter) {
    log.trace("Filtering {} with {}.", Iterables.toString(aspects), filter);
    for (String word : filter) {
      for (final Iterator<Aspect> it = aspects.iterator(); it.hasNext();) {
        final Aspect aspect = it.next();
        if (word.equalsIgnoreCase(aspect.asString())) {
          it.remove();
          log.debug("Removed aspect \"{}\" because it matched \"{}\".", aspect, word);
        }
      }
    }
  }

}
