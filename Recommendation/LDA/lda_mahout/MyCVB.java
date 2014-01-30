package clustering;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.lda.cvb.CVB0Driver;

public class MyCVB {

  public static void main(String[] args) throws ClassNotFoundException, IOException, InterruptedException {
    Configuration conf = new Configuration();
    String inputDirName = "inputDirName";
    Path inputDir = new Path(inputDirName);
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
