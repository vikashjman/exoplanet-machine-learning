ml_section = """
    ---
    ### The Role of Machine Learning in Exoplanet Detection
    Machine learning algorithms have become a pivotal tool in the identification and analysis of exoplanets. 
    By processing vast amounts of data from space telescopes, these algorithms can detect subtle signals 
    of distant planets that may be overlooked by traditional methods.
    
    #### How Machine Learning Enhances Exoplanet Detection
    - **Pattern Recognition:** ML algorithms excel at recognizing patterns in the light curves of stars that 
    indicate a planet transiting in front of them.
    - **Noise Reduction:** They effectively separate the 'noise' of cosmic phenomena from the actual data signals 
    that signify exoplanets.
    - **Scalability:** ML can handle large-scale data analysis, allowing for the simultaneous assessment of data 
    from thousands of stars.
    - **Predictive Analysis:** It can also predict the likelihood of certain characteristics of exoplanets, such 
    as size and orbital period.
    
    Machine learning not only improves detection rates but also helps in characterizing exoplanet atmospheres and 
    potential habitability, pushing the boundaries of astrological science.
    """

what_are_exoplanet = """
        ### What are Exoplanets?
        Exoplanets, or extra-solar planets, are planets that orbit stars beyond our own Sun. 
        They offer a glimpse into the vast variety of planetary systems in our universe and are key 
        to understanding the potential for life elsewhere.
        """
history_of_exoplanet = """
        ### The History of Exoplanet Discovery
        The first confirmed detection of exoplanets came in 1992, and since then, 
        thousands have been discovered. These discoveries have expanded our knowledge of the 
        universe and the potential for habitable worlds beyond Earth.
        """
about_the_data = """
    ---
    ### About the Data
    The data used for exoplanet detection is usually obtained from space observatories, which record the brightness 
    of stars over time. This brightness data, often referred to as light curves, can reveal the presence of planets 
    when they pass in front of a star, causing a temporary dip in brightness.
    
    #### Data Characteristics:
    - **Time Series:** Light curves are time series data that require specialized analysis techniques.
    - **Large Volumes:** Observatories generate large amounts of data that can be challenging to process.
    - **Noise Factors:** Various noise factors such as cosmic events and instrumentation issues need to be accounted for.
    - **Imbalance:** Typically, the number of non-exoplanet samples greatly outnumbers the exoplanet samples, creating 
    class imbalance.
    
    The dataset is derived from the NASA Kepler space telescope's observations, focusing on the flux variations of 
    several thousand stars to identify exoplanets. Each star is labeled as '2' if it has at least one confirmed 
    exoplanet and '1' if there is no confirmed exoplanet.
    
    - **Label '2'**: The star has at least one confirmed exoplanet.
    - **Label '1'**: The star has no confirmed exoplanet.
    
    Planetary transits cause periodic dimming of a star, which can suggest a potential exoplanetary body. These 
    candidate systems require further validation to confirm the presence of exoplanets.
    
    #### Train Set:
    - **Observations (Rows)**: 5087
    - **Features (Columns)**: 3198 (Flux values over time)
    - **Exoplanet-Star Samples**: 37
    - **Non-Exoplanet-Star Samples**: 5050

    #### Test Set:
    - **Observations (Rows)**: 570
    - **Features (Columns)**: 3198 (Flux values over time)
    - **Exoplanet-Star Samples**: 5
    - **Non-Exoplanet-Star Samples**: 565

    The datasets include cleaned data from the Kepler telescope, primarily from Campaign 3, known for its reliable 
    exoplanet findings. Additionally, confirmed exoplanet-stars from other campaigns were added to enrich the dataset.
    Properly processed and analyzed, this data can reveal not only the existence of exoplanets but also provide insights 
    into their properties.
    """


imbalance_warning = """
        **Warning: Significant Class Imbalance Detected**  
        There is a huge disproportion in the dataset: 99.3% represents non-exoplanet stars 
        while only 0.7% corresponds to exoplanet stars. We will be using sampling techniques to balance the data for our analysis.
    """