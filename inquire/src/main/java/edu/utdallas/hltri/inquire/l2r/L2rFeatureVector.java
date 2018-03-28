package edu.utdallas.hltri.inquire.l2r;

import com.google.common.base.CharMatcher;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import edu.utdallas.hltri.ml.Feature;
import edu.utdallas.hltri.ml.vector.SparseFeatureVectorizer;
import edu.utdallas.hltri.ml.vector.VectorUtils;
import edu.utdallas.hltri.util.IntIdentifier;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.ObjIntConsumer;

import edu.umass.cs.mallet.base.types.SparseVector;
import edu.utdallas.hltri.ml.vector.FeatureVector;
import edu.utdallas.hltri.ml.vector.SparseFeatureVector;
import java.util.stream.Stream;
import scala.Char;

/**
 * Created by travis on 11/3/16.
 */
public class L2rFeatureVector implements FeatureVector<Number> {
  public final int topicId;
  public final String comment;
  public final int label;
  public FeatureVector<Number> vector;

  public L2rFeatureVector(int topicId, int label, FeatureVector<Number> vector) {
    this.topicId = topicId;
    this.label = label;
    this.vector = vector;
    this.comment = null;
  }

  public L2rFeatureVector(int topicId, int label, FeatureVector<Number> vector, String comment) {
    this.topicId = topicId;
    this.label = label;
    this.vector = vector;
    this.comment = comment;
  }

  public L2rFeatureVector makeDense() {
    if (vector instanceof SparseFeatureVector) {
      vector = ((SparseFeatureVector<Number>) vector).toDense(0);
    }
    return this;
  }

  public String toSvmRankFormat() {
    return VectorUtils.toSvmRankVector(label, topicId, vector, comment);
  }

  @Override
  public Number getById(int id) {
    return vector.getById(id);
  }

  @Override
  public Number getByName(String name) {
    return vector.getByName(name);
  }

  @Override
  public int[] ids() {
    return vector.ids();
  }

  @Override
  public String[] names() {
    return vector.names();
  }

  @Override
  public int size() {
    return vector.size();
  }

  @Override
  public void forEachWithId(ObjIntConsumer<? super Number> action) {
    vector.forEachWithId(action);
  }

  @Override
  public Iterator<Number> iterator() {
    return vector.iterator();
  }
}
