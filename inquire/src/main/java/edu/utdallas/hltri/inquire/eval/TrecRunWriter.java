/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.utdallas.hltri.inquire.eval;

import com.google.common.base.Preconditions;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 *
 * @author travis
 */
public class TrecRunWriter implements Closeable {
  final private String runtag;
  final private BufferedWriter writer;

  @SuppressWarnings("unused")
  public TrecRunWriter(String path, String runtag) {
    this(Paths.get(path), runtag);
  }

  @SuppressWarnings("unused")
  public TrecRunWriter(File file, String runtag) {
    this(file.toPath(), runtag);
  }

  public TrecRunWriter(Path path, String runtag) {
    Preconditions.checkArgument(Files.notExists(path) || Files.isWritable(path), "path is not writable");
    Preconditions.checkNotNull(runtag, "runtag is null");
    Preconditions.checkArgument(!runtag.isEmpty(), "runtag is empty");
    this.runtag = runtag;
    try {
      this.writer = Files.newBufferedWriter(path, Charset.defaultCharset());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }


  public synchronized void writeResult(Object topic_id, Object doc_id, int rank, double score) {
    writeResultWithTag(topic_id, doc_id, rank, score, this.runtag);
  }

  public synchronized void writeResultWithTag(Object topic_id, Object doc_id, int rank, double score,
      String runtag) {
    try {
      writer.append(topic_id.toString())
          .append("\tQ0\t")
          .append(doc_id.toString()).append('\t')
          .append(Integer.toString(rank)).append('\t')
          .append(Double.toString(score)).append('\t')
          .append(runtag);
      writer.newLine();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override public void close() throws IOException {
    this.writer.close();
  }
}
