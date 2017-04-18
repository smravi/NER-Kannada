package kannada;

/**
 * Created by sruthiravi on 4/13/17.
 */

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Set;
import java.util.regex.Pattern;



import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import edu.uci.ics.crawler4j.crawler.Page;
import edu.uci.ics.crawler4j.crawler.WebCrawler;
import edu.uci.ics.crawler4j.parser.HtmlParseData;
import edu.uci.ics.crawler4j.url.WebURL;


public class MyCrawler extends WebCrawler {
    String crawlStorageFolder = "data/kan/";

    @Override
    public boolean shouldVisit(Page refferringPage,WebURL url) {
        String href = url.getURL().toLowerCase();
        boolean shouldVisit = (href.startsWith("https://kn.wikipedia.org/wiki/") || href.startsWith("http://kn.wikipedia.org/wiki/")); //&& !FILTERS.matcher(href).matches();
        return shouldVisit;
    }

    @Override
    public void visit(Page page) {
        String url = page.getWebURL().getURL();
        System.out.println("URL: " + url);
        if (page.getParseData() instanceof HtmlParseData) {
            HtmlParseData htmlParseData = (HtmlParseData) page.getParseData();
            String text = htmlParseData.getText();
            Document doc = Jsoup.parse(htmlParseData.getHtml());
            doc.select(".mw-content-text").text();
            String html = htmlParseData.getHtml();
            int begin = html.indexOf("<title>");
            int end = html.indexOf("</title>");
            Set<WebURL> links = htmlParseData.getOutgoingUrls();
            System.out.println("Text length: " + text.length());
            System.out.println("Html length: " + html.length());
            System.out.println("Number of outgoing links: " + links.size());
            try {

                Writer out = new BufferedWriter(new OutputStreamWriter(
                        new FileOutputStream(crawlStorageFolder+html.substring(begin + 7, end)+".txt"), "UTF8"));

                doc.select(".toc").remove(); // table of contents remove
                doc.select(".vcard").remove(); // infocard right side summary table remove
                doc.select(".reference").remove(); // remove citations
                doc.select(".thumb").remove(); // remove images
//                Elements elements = doc.select(".mw-headline:contains('இவற்றையும் பார்க்க')").parents();
//                Element parent;
//                for(Element ele: elements){
//                    parent = ele.parent();
//                    parent.siblingElements().remove();
//                    parent.remove();
//                }
                doc.select("sup").remove(); // Remove citation needed

                Elements paras = doc.select("#mw-content-text h2,p");
                for(Element ele: paras){
                    out.write(ele.text().replaceAll("\\(.*?\\) ?", "").replaceAll("\\[.*?\\] ?", "")+"\n");
                }

                out.flush();
                out.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

}

