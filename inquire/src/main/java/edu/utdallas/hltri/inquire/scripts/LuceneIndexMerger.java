package edu.utdallas.hltri.inquire.scripts;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;

import java.nio.file.Path;
import java.nio.file.Paths;

import edu.utdallas.hltri.inquire.lucene.IndexMerger;
import picocli.CommandLine;
import picocli.CommandLine.Option;

public class LuceneIndexMerger {
  @Option(names = {"-s", "--shards", "-i", "--input"},
          required = true,
          arity = "1..*",
          paramLabel = "INDEX-SHARD-DIRECTORY...",
          description = "index shards to be merged")
  Path[] indexShards;

  @Option(names = {"-d", "--dest", "-o", "--output"},
          required = true,
          paramLabel = "MERGED-INDEX-DIRECTORY",
          description = "destination of merged index")
  Path destinationDir;

  @Option(names = {"-a", "--analyzer"},
          paramLabel = "CLASS",
          description = "class of Analyzer to use")
  Analyzer analyzer = new EnglishAnalyzer();

  public static void main(String... args) {
    final LuceneIndexMerger app = new LuceneIndexMerger();
    final CommandLine commandLine = new CommandLine(app)
       .registerConverter(Path.class, Paths::get)
       .registerConverter(Analyzer.class,
           s -> Analyzer.class.cast(Class.forName(s).getConstructor().newInstance())
       );
    commandLine.parse(args);

    final IndexMerger indexMerger = IndexMerger.forShards(app.indexShards);
    indexMerger.mergeTo(app.destinationDir, app.analyzer);
  }
}
