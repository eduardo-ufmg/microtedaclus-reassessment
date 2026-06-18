import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import moa.clusterers.AbstractClusterer;
import moa.clusterers.clustream.WithKmeans;
import moa.clusterers.denstream.WithDBSCAN;
import moa.clusterers.streamkm.StreamKM;

public class MOABenchmark
{
    public static void main(String[] args) throws Exception
    {
        if (args.length < 3) {
            System.err.println("Usage: java MOABenchmark <csvFile> <numAttributes> <Algorithm: "
                               + "CluStream|DenStream|StreamKM>");
            System.exit(1);
        }

        String csvFile = args[0];
        int numAttributes = Integer.parseInt(args[1]);
        String algorithm = args[2];

        // Build schema header — must precede prepareForUse()
        ArrayList<Attribute> attrs = new ArrayList<>();
        for (int i = 0; i < numAttributes; i++) {
            attrs.add(new Attribute("attr" + i));
        }
        Instances dataset = new Instances("stream", attrs, 0);
        InstancesHeader header = new InstancesHeader(dataset);

        AbstractClusterer clusterer;

        switch (algorithm) {
        case "CluStream": {
            WithKmeans cs = new WithKmeans();
            cs.maxNumKernelsOption.setValue(100);
            cs.kOption.setValue(5);
            clusterer = cs;
            break;
        }
        case "DenStream": {
            WithDBSCAN ds = new WithDBSCAN();
            ds.epsilonOption.setValue(0.02);
            ds.muOption.setValue(1.0);
            ds.betaOption.setValue(0.2);
            ds.lambdaOption.setValue(0.001);
            ds.initPointsOption.setValue(100);
            ds.offlineOption.setValue(2.0);
            clusterer = ds;
            break;
        }
        case "StreamKM": {
            StreamKM skm = new StreamKM();
            skm.sizeCoresetOption.setValue(10000);
            skm.numClustersOption.setValue(5);
            skm.lengthOption.setValue(100000);
            clusterer = skm;
            break;
        }
        default:
            throw new IllegalArgumentException("Unknown algorithm: " + algorithm);
        }

        clusterer.setModelContext(header); // must precede prepareForUse()
        clusterer.prepareForUse();

        long startTime = System.nanoTime();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] attValues = new double[numAttributes];
                for (int i = 0; i < numAttributes; i++) {
                    attValues[i] = Double.parseDouble(values[i]);
                }
                DenseInstance inst = new DenseInstance(1.0, attValues);
                inst.setDataset(dataset); // binds schema; resolves NPE
                clusterer.trainOnInstance(inst);
            }
        }

        long endTime = System.nanoTime();
        System.out.println("Execution Time (ms) [" + algorithm +
                           "]: " + (endTime - startTime) / 1_000_000);
    }
}
