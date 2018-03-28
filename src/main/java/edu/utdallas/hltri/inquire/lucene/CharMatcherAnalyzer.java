/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.utdallas.hltri.inquire.lucene;

import com.google.common.base.CharMatcher;

import org.apache.lucene.analysis.Analyzer;

/**
 *
 * @author travis
 */
public class CharMatcherAnalyzer extends Analyzer {
  private final CharMatcher matcher;

  public CharMatcherAnalyzer(CharMatcher matcher) {
    super();
    this.matcher = matcher;
  }


  @Override
  protected TokenStreamComponents createComponents(String fieldName) {
    return new TokenStreamComponents(new CharMatcherTokenizer(matcher));
  }


}
