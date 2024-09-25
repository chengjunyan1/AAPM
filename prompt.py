news_analysis='''You are a helpful assistant designed to analyze the business news and output JSON.
You need to extract the following information from the news:
1. Tickers: The stock tickers of companies most closely related to the news. If there is no relevant ticker, return an empty list. You should never make up a ticker that does not exist.
2. Topics: The topics of the news, such as "earnings", "merger", "lawsuit", etc.
3. Content: Use brief language to describe the key information and preserve the data from the news.
Now, analyze the following news and output the JSON file:

{news}
'''

news_reason='''You are a helpful assistant designed to analyze the business news to assist portfolio management.
Now, read this latest news and summarize it in one single paragraph, preserving data, datetime of the events, and key information, and include new insights for investment using the recommended relevant information:

{news}
'''


news_dialog_begin='''You are a helpful assistant designed to analyze the business news to assist portfolio management. 
You will help me analyze this latest news from The Wall Street Journal and provide an analysis report, then I will search the relevant news or articles from the knowledge base based on your analysis report to help you refine it iteratively in multiple rounds. 
Let's start with this latest news, provide your analysis report, and I will help you refine it with the relevant information later, if you think this news is completely not helpful for investment now or future, call skip function to skip it, do not skip it if it may contain helpful information to future investment: 

{inputs}

Here is a summary of the macroeconomics by today and the investment notes:

{macro}
'''

news_dialog_cont='''Based on your current analysis report, I found those potentially relevant news and excerpts from the knowledge base, please refine your analysis report with this information:

{inputs}
'''

news_dialog_end='''Based on your current analysis report, I found those potentially relevant news and excerpts from the knowledge base, now finish your analysis report with them:

{inputs}
'''

macro_init='''By September 2021, the global macroeconomic landscape was heavily influenced by the ongoing impacts of the COVID-19 pandemic. Many countries were in various stages of recovery, grappling with challenges such as disrupted supply chains, inflationary pressures, and shifts in employment patterns. Key points include:
1. **Economic Recovery**: Different regions experienced uneven recovery, with some economies bouncing back faster due to successful vaccination campaigns and substantial fiscal stimuli. For instance, the U.S. and China showed signs of robust economic rebound, whereas many European countries were still struggling with economic output below pre-pandemic levels.
2. **Inflation Concerns**: Rising inflation became a significant concern in many countries, partly due to supply chain disruptions and increased demand as economies reopened. This led to higher prices for commodities, goods, and services.
3. **Monetary Policy**: Central banks, including the U.S. Federal Reserve and the European Central Bank, maintained accommodative monetary policies, with low interest rates to support economic growth. However, there was growing discourse about when and how to start tapering these measures.
4. **Employment Fluctuations**: While some sectors and countries saw a rapid recovery in employment levels, others faced ongoing job losses, highlighting the pandemic's uneven impact across different industries.
5. **Supply Chain Disruptions**: Global supply chains were strained, impacting everything from consumer electronics to automobile manufacturing, leading to shortages and delays.
6. **Shifts in Consumer Behavior**: The pandemic accelerated trends like online shopping and remote working, reshaping economic activities and consumer behaviors in lasting ways.
Overall, the state of global macroeconomics by September 2021 was defined by recovery efforts amidst ongoing challenges, with significant variability between different countries and regions.
'''

macro_update='''
Here is the current summary of the macroeconomic landscape and investment notes as of {date}:

{macro}

Now, given the latest news and the analysis report, update the macroeconomic summary with the new insights and impacts from the news. Include any relevant information that could influence the global economic outlook, such as geopolitical events, policy changes, or economic indicators. 
You should also take note of any important notes about investment trend and chances. Here are the latest news and the analysis report:

{news}

Now, update the macroeconomic summary with the new insights and impacts from the news as well as the investment notes.
'''



