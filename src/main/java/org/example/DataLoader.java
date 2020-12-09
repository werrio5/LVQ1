package org.example;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author werrio5
 */
public class DataLoader {

    private static Map<Integer[],String> trainData;
    private static Map<Integer[],String> testData;
    private static int classCount;
    private static File[] directories;

    public static Map<Integer[],String> getTrainData(){

        //если не загруженны - загрузить
        if(trainData == null){
            loadData();
        }

        return trainData;
    }

    public static Map<Integer[],String> getTestData(){
        return testData;
    }

    public static Set<String> getTestNames(){
        Set<String> names = new HashSet<>(trainData.values());
        return names;
    }

    /**
     * данные масштабируются от 0..63 до 0..255
     */
    private static void loadData() {
        //путь к данным
        String curDir = System.getProperty("user.dir");
        String datapath = curDir + File.separator + "data";
        File folder = new File(datapath);

        //список папок
        directories = folder.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File current, String name) {
                return new File(current, name).isDirectory();
            }
        });

        //data
        trainData = new HashMap<>();
        testData = new HashMap<>();
        classCount = directories.length;


        //перебор папок
        for (File directory : directories) {
            //файлы
            File[] files = directory.listFiles();
            if(files == null) continue;

            //перебор файлов
            //все файлы, кроме последнего в train
            //последний в test
            for (File file : files) {
                //файл
                try {
                    BufferedReader br = new BufferedReader(new FileReader(file));
                    //чтение строк
                    try {
                        String line = br.readLine();
                        while (line != null) {
                            //разделить строку
                            String[] digitsStrings = line.split(" ");
                            Integer[] digits = new Integer[digitsStrings.length];

                            //str to int
                            // 4 -> scale 63 to 255
                            for (int i = 0; i < digits.length; i++) {
                                digits[i] = 4 * Integer.valueOf(digitsStrings[i]);
                            }

                            //запись
                            //String className = replaceClass(directory.getName());
                            String className = directory.getName();
                            if(file.equals(files[files.length - 1]))
                                testData.put(digits,className);
                            else
                                trainData.put(digits,className);

                            //следующая строка
                            line = br.readLine();
                        }
                    } catch (IOException ex) {
                        Logger.getLogger(DataLoader.class.getName()).log(Level.SEVERE, null, ex);
                    }
                } catch (FileNotFoundException ex) {
                    Logger.getLogger(DataLoader.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }

        filterData(directories);
    }

    private static void filterData(File[] directories){
        //all data
        Map<Integer[],String> fullData = new HashMap<>();
        fullData.putAll(trainData);

        Boolean[][][] valuesRegistered = new Boolean[64][64][64];
        //unique lines
        Set<Integer[]> lineSet = new HashSet<>();

        for(Integer[] line:fullData.keySet()){
            if(valuesRegistered[line[0]/4][line[1]/4][line[2]/4] == null){
                lineSet.add(line);
                valuesRegistered[line[0]/4][line[1]/4][line[2]/4] = new Boolean(true);
            }
            else{
                valuesRegistered[line[0]/4][line[1]/4][line[2]/4] = false;
            }
        }

        //delete excess
        List<Integer[]> lineSetCopy = new ArrayList<>();
        lineSetCopy.addAll(lineSet);
        for(Integer[] line: lineSetCopy){
            if(valuesRegistered[line[0]/4][line[1]/4][line[2]/4] != null){
                if(valuesRegistered[line[0]/4][line[1]/4][line[2]/4] == false){
                    lineSet.remove(line);
                }
            }
        }

        Map<Integer[],String> filteredTrainData = new HashMap<>();
        //define class
        for(Integer[] line: lineSet){
            filteredTrainData.put(line, fullData.get(line));
        }

        trainData = filteredTrainData;

        Map<Integer[],String> fullTestData = new HashMap<>();
        fullTestData.putAll(testData);

        Boolean[][][] testValuesRegistered = new Boolean[64][64][64];
        //unique lines
        Set<Integer[]> testLineSet = new HashSet<>();
        for(Integer[] line:fullTestData.keySet()){

            if(testValuesRegistered[line[0]/4][line[1]/4][line[2]/4] == null){
                testLineSet.add(line);
                testValuesRegistered[line[0]/4][line[1]/4][line[2]/4] = new Boolean(true);
            }
            else{
                testValuesRegistered[line[0]/4][line[1]/4][line[2]/4] = false;
            }
        }

        //delete excess
        List<Integer[]> testLineSetCopy = new ArrayList<>();
        lineSetCopy.addAll(testLineSet);
        for(Integer[] line: testLineSetCopy){
            if(testValuesRegistered[line[0]/4][line[1]/4][line[2]/4] != null){
                if(testValuesRegistered[line[0]/4][line[1]/4][line[2]/4] == false){
                    testLineSet.remove(line);
                }
            }
        }

        Map<Integer[],String> filteredTestData = new HashMap<>();
        //define class
        for(Integer[] line: testLineSet){
            filteredTestData.put(line, fullTestData.get(line));
        }

        testData = filteredTestData;
    }

    public static double[][] initWeights(){
        double[][] weights = new double[3][classCount];
        for(int i=0;i<directories.length;i++){
            for(Integer[] line:trainData.keySet()){
                if(trainData.get(line).equals(directories[i].getName())){
                    weights[0][i]=line[0];
                    weights[1][i]=line[1];
                    weights[2][i]=line[2];
                    break;
                }
            }
        }
        return weights;
    }

    private static String replaceClass(String sourceClass){
        switch (sourceClass){
            case "Climb_stairs":
            case "Descend_stairs":
                return "Stairs";

            case "Standup_chair":
            case "Sitdown_chair":
                return "chair";

            case "Getup_bed":
            case "Liedown_bed":
                return "bed";

            case "Drink_glass":
            case "Eat_soup":
            case "Eat_meat":
                return "eat";

            default:
                return sourceClass;
        }
    }
    
    public static void Save(StringBuilder sb){
        String curDir = System.getProperty("user.dir");
        String datapath = curDir + File.separator + "processing" + File.separator;
        String filename = "res2.csv";
        File file = new File(datapath+filename);
        try {
            Files.deleteIfExists(file.toPath());
            PrintWriter f = new PrintWriter(file);
            f.write(sb.toString());
            f.close();
        } catch (IOException ex) {
            Logger.getLogger(DataLoader.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}

