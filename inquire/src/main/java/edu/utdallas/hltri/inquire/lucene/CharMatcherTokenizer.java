/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.utdallas.hltri.inquire.lucene;

import com.google.common.base.CharMatcher;

import org.apache.lucene.analysis.util.CharTokenizer;

/**
 *
 * @author travis
 */
public class CharMatcherTokenizer extends CharTokenizer {
  private final CharMatcher matcher;

  CharMatcherTokenizer(CharMatcher matcher) {
    this.matcher = matcher;
  }

  @Override protected boolean isTokenChar(int c) {
    return !matcher.matches((char) c);
  }
}
