package clustering;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.clustering.lda.cvb.CVB0Driver;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;

public class ClusterWithCVB {

  public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
    // parameters
    int minSupport = 2;
    int minDf = 2;
    int maxDFPercent = 90;
    int maxNGramSize = 2;
    int minLLRValue = 50;
    int reduceTasks = 1;
    int chunkSize = 200;
    int norm = 2;
    boolean sequentialAccessOutput = true;
    boolean namedVectors = true;
    
    // directory of doc sequence file(s)
    //String inputDirName = args[0];
    String inputDirName = "yelp_seqfile";
    Path inputDir = new Path(inputDirName);

    // directory where clusters will be written
    String outputDirName = "yelp_output";
    Path outputDir = new Path(outputDirName);

    Configuration conf = new Configuration();

    HadoopUtil.delete(conf, outputDir);

    // converts input docs in sequence file format in input_dir into token
    // array in output_dir/tokenized-documents
    Path tokenizedPath = new Path(outputDirName,
        DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
    MyAnalyzer analyzer = new MyAnalyzer();
    DocumentProcessor.tokenizeDocuments(inputDir, analyzer.getClass()
        .asSubclass(Analyzer.class), tokenizedPath, conf);

    // reads token array in output_dir/tokenized-documents and writes term
    // frequency vectors in output_dir (under tf-vectors)
    String tfDirName = DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER;
    DictionaryVectorizer.createTermFrequencyVectors(tokenizedPath,
        outputDir, tfDirName, conf, minSupport, maxNGramSize,
        minLLRValue, -1.0f, false, reduceTasks, chunkSize,
        sequentialAccessOutput, namedVectors);

    Path sequenceDir = new Path(inputDirName);
    String outputTopicDirName = "outputTopicDirName";
    Path outputTopicPath = new Path(outputTopicDirName);
    int numTopics = 10;
    int numTerms = 8000;
    double alpha = 5.0;
    double eta = 0.5;
    int maxIteration = 20;
    int iterationBlockSize = 50;
    double convergenceDelta = 10;
    Path dictPath = new Path("dictPath");
    Path docTopicOutputPath = new Path("docTopicOutputPath");
    Path topicModelStateTempPath = new Path("topicModelStateTempPath");
    long randomSeed = 0;
    float testFraction = (float) 0.5;
    int numTrainThreads = 1;
    int numUpdateThreads = 1;
    int maxIterasPerDoc = 20;
    int numRedcueTasks = 10;
    boolean backfillPerplexity = false;

    CVB0Driver.run(conf, inputDir,
        outputTopicPath,
        numTopics,
        numTerms, 
        alpha, 
        eta, 
        maxIteration, 
        iterationBlockSize,
        convergenceDelta, 
        dictPath, 
        docTopicOutputPath, topicModelStateTempPath, randomSeed, testFraction, 
        numTrainThreads, numUpdateThreads, maxIterasPerDoc,
        numRedcueTasks, backfillPerplexity);
  }
}
