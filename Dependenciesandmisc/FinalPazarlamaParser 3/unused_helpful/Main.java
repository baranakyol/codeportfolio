import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.StringReader;
import java.net.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import org.jsoup.Connection;
import org.jsoup.Connection.Method;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;


public class Main {

	/**
	 * @param args
	 */
	
	private static String cookie;
	
	public static void main(String[] args) throws FileNotFoundException {
		// TODO Auto-generated method stub
		boolean loggedIn = logon();

		Scanner scan = new Scanner(new File("sample_isbn.txt"));
		
		ArrayList<String> ISBNs = new ArrayList<String>();

		int i=1;
		while(scan.hasNextLine() && i==1)
		{
			//if(i==5)
				ISBNs.add(scan.next());
			//else
				scan.next();
			i++;
		}
		scan.close();

		PrintWriter writer;
		writer = new PrintWriter(new File("stock.txt"));
		writer.print("ISBN\tDescription\tFinalProductID\tstock\tlink\n");

		if(loggedIn)
		{
			for(String isbn:ISBNs)
			{
				Map<String, String> book_details = getProduct(getStock(isbn));
				String link = book_details.get("link");
				String stock = book_details.get("stok");
				String finalProductId = book_details.get("FinalProductID");
				String bookDescription = book_details.get("Book");
				writer.print(isbn+"\t"+bookDescription+"\t"+finalProductId+"\t"+stock+"\t"+link+"\n");

			}
		}
		writer.close();
	}
	public static String getStock(String keyword)
	{
		boolean flag = false;
		String response = "";
		try {
		    // Construct data
		    String data = URLEncoder.encode("FilterType", "UTF-8") + "=" + URLEncoder.encode("0", "UTF-8");
		    data += "&" + URLEncoder.encode("Keyword", "UTF-8") + "=" + URLEncoder.encode(keyword, "UTF-8");

		    // Send data
		    URL url = new URL("https://www.finalpazarlama.com/arama?"+data);
		    URLConnection conn = url.openConnection();
		    conn.setDoOutput(true);
		    OutputStreamWriter wr = new OutputStreamWriter(conn.getOutputStream());
		    //wr.write(data);
		    wr.flush();

		    // Get the response
		    BufferedReader rd = new BufferedReader(new InputStreamReader(conn.getInputStream()));
		    String line;

		    while ((line = rd.readLine()) != null) {
		    	// Process line...
		    	if(line.contains("urunResim"))
		    		response = rd.readLine();
		    	else if(line.contains("Kayıt Bulunamadı."))
		    		flag= true;
		    }

            
		    wr.close();
		    rd.close();
		} catch (Exception e) {
			
		}
		
		if(!flag)
		{
			response = response.split("/")[4];
			response = response.split("\"")[0];		
		}else
			response = "badISBN";


		
		return response;
	}
	public static Map<String, String> getProduct(String keyword)
	{
		Map<String, String> response = new HashMap<String, String>();
	    String stock = null;
	    String image_link = null;
	    String book_description = null;
	    String price = null;
	    String discounted_price = null;
	    
	    if(keyword != "badISBN")
	    {
			try {
			    // Construct data
			    String data = URLEncoder.encode("productID", "UTF-8") + "=" + URLEncoder.encode(keyword, "UTF-8");
			    // Send data
			    URL url = new URL("https://www.finalpazarlama.com/Product/Detail?"+data);
			    URLConnection conn = url.openConnection();
			    conn.setRequestProperty("Cookie", cookie);
	
			    // Get the response
			    BufferedReader rd = new BufferedReader(new InputStreamReader(conn.getInputStream()));
			    String line;
			    while ((line = rd.readLine()) != null) {
			        // Process line...
			    	if(line.contains("class=\"fancybox"))
			    	{
			    		image_link = rd.readLine();		    		
			    	}
			    	if(line.contains("detayD sag"))
			    		stock = rd.readLine();			    	
			    	if(line.contains("dRib\""))
			    	{
			    		book_description = rd.readLine();	    	
			    		book_description = rd.readLine();	    	
			    		book_description = rd.readLine();	    	
			    	}
			    	
			    }
			    //wr.close();
			    rd.close();
			} catch (Exception e) {
				
			}
	
			try
			{
				image_link = image_link.split("\"")[1];
			}
			catch(Exception e)
			{
				
			}
			
			if(image_link!=null && stock != null)
			{
				
				response.put("link", image_link);
				response.put("stok", stock.trim());			
				response.put("FinalProductID", keyword);			
				response.put("Book", book_description);			
			}
	    }else
	    {
			response.put("link", keyword);
			response.put("stok", keyword);			
			response.put("FinalProductID", keyword);	
	    }
		return response;
	}

	public static boolean logon()
	{
		boolean flag = false;
		
		try {
		    // Construct data
		    String data = URLEncoder.encode("CustomerCode", "UTF-8") + "=" + URLEncoder.encode("M05435", "UTF-8");
		    data += "&" + URLEncoder.encode("UserName", "UTF-8") + "=" + URLEncoder.encode("akyolbaran1@gmail.com", "UTF-8");
		    data += "&" + URLEncoder.encode("Password", "UTF-8") + "=" + URLEncoder.encode("amp14mes", "UTF-8");
		    data += "&" + URLEncoder.encode("X-Requested-With", "UTF-8") + "=" + URLEncoder.encode("XMLHttpRequest", "UTF-8");
		    
		    System.setProperty("http.agent", "");
		    // Send data
		    URL url = new URL("https://www.finalpazarlama.com/Account/LogOn");
		    URLConnection conn = url.openConnection();
		    conn.setRequestProperty("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:41.0) Gecko/20100101 Firefox/41.0");
		    conn.setRequestProperty("X-Requested-With", "XMLHttpRequest");
		    conn.setRequestProperty("Referer", "https://www.finalpazarlama.com/");
		    conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8");
		    conn.setDoOutput(true);
		    
	    
		    OutputStreamWriter wr = new OutputStreamWriter(conn.getOutputStream());
		    wr.write(data);
		    wr.flush();
		    
	
		    String headerName=null;
		    for (int i=1; (headerName = conn.getHeaderFieldKey(i))!=null; i++) {
		     	if (headerName.equals("Set-Cookie")) {
		     		cookie = conn.getHeaderField(i);
		     	}
		    }
		
		    // Get the response
		    BufferedReader rd = new BufferedReader(new InputStreamReader(conn.getInputStream()));
		    String line;
		    while ((line = rd.readLine()) != null) {
		        // Process line...
		    	if(line.contains("\"IsSuccess\":true"))
		    		flag= true;
		    }
		    
		    
		    wr.close();
		    rd.close();
		} catch (Exception e) {
			
		}
		
		return flag;
	}
}
