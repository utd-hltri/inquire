package edu.utdallas.hltri.inquire.lucene;

import org.apache.lucene.util.Bits;

/**
 * Created by travis on 8/4/14.
 */
public class SingleDocumentBits implements Bits {
  final int length;

  int target;

  public SingleDocumentBits(int length, int target) {
    this.length = length;
    this.target = target;
  }

  @Override public boolean get(final int index) {
    return index == target;
  }

  @Override public int length() {
    return length;
  }
}
