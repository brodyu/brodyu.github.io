---
layout: post
title:  "Jekyll Tutorial"
date:   2020-03-08 18:05:55 +0300
image:  '/assets/img/webdev.jpg'
tags:   Jekyll
---
<p>Trying to create a website is complicated. As if developing the front and backend wasn't arduous enough, one has to worry about security and network issues as well which discourages many from pursuing their own website. In this tutorial, I am going to provide you with an alternative for simple static webpages that is completely free.</p>

<p>For this tutorial we are going to be using Jekyll, a free and open source static site generator that runs on the Ruby programming language. What makes Jekyll great is that instead of databases, Jekyll renders Markdown templates, which makes writing HTML a whole lot easier. Moreover, Jekyll works well in combination with front-end frameworks such as Semantic UI and Bootstrap. For deployment of our Jekyll webpage, we are going to use GitHub Pages to host our websites for free.</p>

<p>There are two ways to create webpages using Jekyll. The first is most conventional, while the second is much easier and faster. I will be going over both in this tutorial.</p>

<h2 class="section-heading">Installation</h2>

<p>First, head over to the <a href="https://jekyllrb.com/docs/installation/">Jekyll documentation page</a> and follow the installation instructions for your specific operating system.</p>

<h2 class="section-heading">Creating a Site</h2>

<p>Open up your terminal and type:</p>

<code>jekyll new name_of_site</code>

<p>Where name_of_site is the name of the website you want to create.<p>

<p>You should see a new repository appear that Jekyll automatically created with its base template. Now move into the repository and type:</p>

<code>bundle exec jekyll serve</code>

<p>Bundle exec needs to be included in this command the first time you run your website. After, you only need to type jekyll serve. This command starts a local host website on port 4000. Now you should be able to view your website locally by going into your preferred web browser and type: localhost:4000</p>

<p>Congratulations! You just created your first website. From here you can write your own blog posts, change to a different theme, change the layout, etc. I will include a more in-depth tutorial at the end for messing around with all these different options.</p>

<p>When I created my first site utilizing this method, I was quickly bothered by the mundane default theme, minima. However, while trying to convert my existing website into another theme, I ran into a lot of problems.</p>

<h2 class="section-heading">The Easy Way</h2>

<p>A much more simplified way of doing all this and troubleshooting future theme problems is to just fork the theme you want from GitHub. For this website, I forked BlackrockDigital’s <a href="https://github.com/BlackrockDigital/startbootstrap-clean-blog-jekyll">startbootstrap-clean-blog-jekyll</a>. After you fork the repository you want, delete the gh-pages branch.</p>

<p>Next, create a new branch called gh-pages and update the default branch from master to gh-pages. We must delete their gh-pages branch because they have a custom domain name that will not work for us.</p>

<p>Now we can go to our GitHub pages site in Settings to view our hosted gh-page.</p>

<img class="img-fluid" src="https://guides.github.com/features/pages/launch-theme-chooser.png">

<p>If you receive a problem building your site due to the use to relative_permalinks, all you have to do is delete this line from the config.yml file:</p>

<code>relative_permalinks: true</code>

<p>Finally, you are basically done. From here, you can edit the config.yml file to edit the main information of your webpage, or you can add your own blog posts in the posts folder. I suggest heading over to <a href="https://www.youtube.com/watch?v=T1itpPvFWHI&list=PLLAZ4kZ9dFpOPV5C5Ay0pHaa0RJFhcmcB">Mike Dane's youtube channel</a> for more detailed tutorials on editing your website.</p>

<p>Photographs by <a href="https://unsplash.com/">Unsplash</a>.</p>
