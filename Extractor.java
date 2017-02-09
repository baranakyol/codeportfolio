import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import org.jsoup.Connection;
import org.jsoup.Connection.Method;
import org.jsoup.Connection.Response;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.select.Elements;



public class Extractor {

	/**
	 * @param args
	 */
	private static String cookie;
	private static ArrayList<String> ISBNs;
	
	public static void main(String[] args) throws FileNotFoundException {
		// TODO Auto-generated method stub
		cookie = "";
		boolean loggedIn = logIn();
		readISBNsFromFile();
		
		PrintWriter writer;
		writer = new PrintWriter(new File("stock.txt"));
		writer.print("ISBN\tDescription\tPublisher\tStock\tImage Path\tNet Price\tTax Rate\tGross Price\tDiscount Rate\tDiscounted Net Price\tTaxed Net Price\n");

		if(loggedIn)
		{
			for(String isbn:ISBNs)
			{	
				System.out.println("Current ISBN: "+isbn);
				Map<String, String> book_details = productDetails(isbn);
				String description = book_details.get("description");
				String publisher = book_details.get("publisher");
				String stock = book_details.get("stock");
				String img_path = book_details.get("img_path");
				String net_price = book_details.get("net_price");
				String tax_rate = book_details.get("tax_rate");
				String gross_price = book_details.get("gross_price");
				String discount_rate = book_details.get("discount_rate");
				String discounted_net_price = book_details.get("discounted_net_price");
				String taxed_net_price = book_details.get("taxed_net_price");

				writer.print(isbn+"\t"+description+"\t"+publisher+"\t"+stock+"\t"+img_path+
								  "\t"+net_price+"\t"+tax_rate+"\t"+gross_price+"\t"+discount_rate+
								  "\t"+discounted_net_price+"\t"+taxed_net_price+"\n");

			}
		}
		writer.close();
		System.out.println("Complete :)");

	}

	private static boolean logIn() {
		boolean flag = false;
		
		Connection.Response res;
		try {
			res = Jsoup.connect("https://www.finalpazarlama.com/Account/LogOn")
				    .data("CustomerCode", "M05435", "UserName", "akyolbaran1@gmail.com", "Password", "amp14mes", "X-Requested-With", "XMLHttpRequest")
				    .userAgent("Mozilla")
				    .method(Method.POST)
				    .ignoreContentType(true)
				    .execute();
			
			Document doc = res.parse();
			cookie = res.cookie("FINALSID");
			String loginResponse = doc.body().text();
			if(loginResponse.contains("\"IsSuccess\":true"))
				flag = true;			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
			
		return flag;
	}

	private static void readISBNsFromFile() throws FileNotFoundException
	{
		Scanner scan = new Scanner(new File("sample_isbn.txt"));
		ISBNs = new ArrayList<String>();
		while(scan.hasNextLine())
		{
			ISBNs.add(scan.next());
		}
		scan.close();

	}
	
	private static Map<String, String> productDetails(String ISBN)
	{
		HashMap details = new HashMap<String,String>();
		String finalCode, description ,publisher, stock, img_path,
			   net_price, tax_rate , gross_price, discount_rate, discounted_net_price, taxed_net_price="";
		try {
			Document doc = Jsoup.connect("https://www.finalpazarlama.com/arama?")
					  .data("FilterType", "0", "Keyword", ISBN)
					  .userAgent("Mozilla")
					  .cookie("FINALSID", "7dc4bb9c-b36b-4a0d-b6f9-84df2fe4218f")
					  .timeout(3000)
					  .get();

			Element dOrta = doc.getElementById("dOrta");
			Element dStokDetay = dOrta.children().first();
			Element sdSol_sol = dStokDetay.children().first();
			
			Element desc = sdSol_sol.getElementsByClass("dRib").first();
			if(desc != null)
			{
				description = desc.text();
				details.put("description", description);				
			
			Element fancybox = sdSol_sol.getElementsByClass("resim").first();
			img_path = fancybox.getElementsByTag("a").attr("href");
			details.put("img_path", img_path);
			
			Element detayD_1 = sdSol_sol.getElementsByClass("detayD").first();
			publisher = detayD_1.text();
			details.put("publisher", publisher);
			
			Elements stockStatus = sdSol_sol.select("h4:matchesOwn(Stok Durumu)");
			stock = ((Element)stockStatus.first().parent().nextSibling().nextSibling()).text();
			details.put("stock", stock);				

			Element sdSag_sag = dStokDetay.children().get(1);
			Element el1 = sdSag_sag.getElementsByTag("li").get(1);
			net_price = el1.text(); 
			net_price = net_price.split(":")[1];
			details.put("net_price", net_price);
			
			Element el2 = sdSag_sag.getElementsByTag("li").get(2);
			tax_rate = el2.text(); 
			tax_rate = tax_rate.split(":")[1];
			details.put("tax_rate", tax_rate);
			
			Element el3 = sdSag_sag.getElementsByTag("li").get(3);
			gross_price = el3.text(); 
			gross_price = gross_price.split(":")[1];
			details.put("gross_price", gross_price);
			
			Element el4 = sdSag_sag.getElementsByTag("li").get(4);
			discount_rate = el4.text(); 
			discount_rate = discount_rate.split(":")[1];
			details.put("discount_rate", discount_rate);
			
			Element el5 = sdSag_sag.getElementsByTag("li").get(5);
			discounted_net_price = el5.text(); 
			discounted_net_price = discounted_net_price.split(":")[1];
			details.put("discounted_net_price", discounted_net_price);
			
			Element el6 = sdSag_sag.getElementsByTag("li").get(6);
			taxed_net_price = el6.text();
			taxed_net_price = taxed_net_price.split(":")[1];
			details.put("taxed_net_price", taxed_net_price);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
		}
		return details;
	}	
}
