package kannada;

/**
 * Created by sruthiravi on 4/13/17.
 */
import edu.uci.ics.crawler4j.crawler.CrawlConfig;
import edu.uci.ics.crawler4j.crawler.CrawlController;
import edu.uci.ics.crawler4j.fetcher.PageFetcher;
import edu.uci.ics.crawler4j.robotstxt.RobotstxtConfig;
import edu.uci.ics.crawler4j.robotstxt.RobotstxtServer;


public class Controller {
    public static void main(String[] args) throws Exception {
        String crawlStorageFolder = "data/kan/";
        int numberOfCrawlers = 7;
        CrawlConfig config = new CrawlConfig();
        config.setCrawlStorageFolder(crawlStorageFolder);
        config.setMaxDepthOfCrawling(5);
        config.setMaxPagesToFetch(100000);
        config.setIncludeBinaryContentInCrawling(true);
        config.setMaxDownloadSize(Integer.MAX_VALUE);
        config.setPolitenessDelay(300);
        PageFetcher pageFetcher = new PageFetcher(config);
        RobotstxtConfig robotstxtConfig = new RobotstxtConfig();
        RobotstxtServer robotstxtServer = new RobotstxtServer(robotstxtConfig, pageFetcher);
        CrawlController controller = new CrawlController(config, pageFetcher, robotstxtServer);

        controller.addSeed("https://kn.wikipedia.org/wiki/%E0%B2%B5%E0%B2%B0%E0%B3%8D%E0%B2%97:%E0%B2%87%E0%B2%A4%E0%B2%BF%E0%B2%B9%E0%B2%BE%E0%B2%B8");

        controller.start(MyCrawler.class, numberOfCrawlers);

        controller.shutdown();
    }
}