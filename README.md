# mc-tk
A Monte-Carlo toolkit for educational purposes.

> pip install mc-tk

# package architecture

    experiments
        - classical / typical experiments in probability

    distributions 
        - inclulde MC experiments that produce common distributions

    samplings
        - sampling distributions of statistic used in hypothesis tests

# functions
<div>
<table border="1" cellspacing="0">
<tbody>
<tr>
<td width="166">
<p>Module</p>
</td>
<td width="170">
<p>Function</p>
</td>
<td width="512">
<p>Description</p>
</td>
</tr>
<tr>
<td rowspan="7" width="166">
<p>mc.experiments</p>
</td>
<td width="170">
<p>pi()</p>
</td>
<td width="512">
<p>Perform Buffon&rsquo;s needle experiment to estimate&nbsp;&pi; .</p>
</td>
</tr>
<tr>
<td width="170">
<p>parcel()</p>
</td>
<td width="512">
<p>Simulate&nbsp;a&nbsp;bi-directional&nbsp;parcel&nbsp;passing&nbsp;game.</p>
</td>
</tr>
<tr>
<td width="170">
<p>dices()</p>
</td>
<td width="512">
<p>Estimate&nbsp;the&nbsp;probabilities&nbsp;of&nbsp;various&nbsp;dice&nbsp;combinations.</p>
</td>
</tr>
<tr>
<td width="170">
<p>prisoners()</p>
<p>Prisoners_limit()</p>
</td>
<td width="512">
<p>The&nbsp;famous&nbsp;locker&nbsp;puzzle(100-prisoner&nbsp;quiz). This&nbsp;function&nbsp;will&nbsp;prove&nbsp;that&nbsp;the&nbsp;survival&nbsp;chance&nbsp;limit&nbsp;is&nbsp;1&minus;ln2 when&nbsp;n&nbsp;approaches&nbsp;+&infin; .</p>
</td>
</tr>
<tr>
<td width="170">
<p>galton_board()</p>
</td>
<td width="512">
<p>Use&nbsp;the&nbsp;classic&nbsp;Galton&nbsp;board&nbsp;experiment&nbsp;to&nbsp;produce&nbsp;a&nbsp;binomial&nbsp;distribution.</p>
</td>
</tr>
<tr>
<td width="170">
<p>paper_clips()</p>
</td>
<td width="512">
<p>Use&nbsp;the&nbsp;paper&nbsp;clip&nbsp;experiment&nbsp;to&nbsp;produce&nbsp;a&nbsp;Zipf&nbsp;distribution.</p>
</td>
</tr>
<tr>
<td width="170">
<p>sudden_death()</p>
</td>
<td width="512">
<p>This&nbsp;function&nbsp;simulates&nbsp;a&nbsp;sudden&nbsp;death&nbsp;game&nbsp;to&nbsp;produce the&nbsp;exponential&nbsp;distribution.</p>
</td>
</tr>
<tr>
<td rowspan="2" width="166">
<p>mc.distributions</p>
</td>
<td width="170">
<p>poisson()</p>
</td>
<td width="512">
<p>This&nbsp;function&nbsp;will&nbsp;demonstrate&nbsp;that&nbsp;Poisson&nbsp;is&nbsp;a&nbsp;limit&nbsp;distribution&nbsp;of b(n,p) when&nbsp;n&nbsp;is&nbsp;large, and&nbsp;p&nbsp;is&nbsp;small.</p>
</td>
</tr>
<tr>
<td width="170">
<p>benford()</p>
</td>
<td width="512">
<p>Verify&nbsp;Benford&rsquo;s&nbsp;law&nbsp;using&nbsp;real-life&nbsp;datasets, including&nbsp;the&nbsp;stock market&nbsp;data, international&nbsp;trade&nbsp;data, and&nbsp;the&nbsp;Fibonacci&nbsp;series.</p>
</td>
</tr>
<tr>
<td rowspan="17" width="166">
<p>mc.samplings</p>
</td>
<td rowspan="8" width="170">
<p>clt()</p>
</td>
<td width="512">
<p>Using&nbsp;various&nbsp;underlying&nbsp;distributions&nbsp;to&nbsp;verify&nbsp;the&nbsp;central&nbsp;limit&nbsp;&nbsp;theorem.&nbsp;This&nbsp;function&nbsp;provides&nbsp;the&nbsp;following&nbsp;underlying&nbsp;distributions.</p>
</td>
</tr>
<tr>
<td width="512">
<p>&rsquo;uniform&rsquo; - a&nbsp;uniform&nbsp;distribution&nbsp;U(-1,1).</p>
</td>
</tr>
<tr>
<td width="512">
<p>&rsquo;expon&rsquo;- an&nbsp;exponential distribution Expon(1).</p>
</td>
</tr>
<tr>
<td width="512">
<p>&rsquo;poisson&rsquo; - poisson&nbsp;distribution &pi;(1).</p>
</td>
</tr>
<tr>
<td width="512">
<p>&rsquo;coin&rsquo;- Bernoulli&nbsp;distribution&nbsp;with&nbsp;p&nbsp;= 0.5.</p>
</td>
</tr>
<tr>
<td width="512">
<p>&rsquo;tampered_coin&rsquo;&nbsp;- PMF:{0:0.2,1:0.8}, i.e., head&nbsp;more&nbsp;likely&nbsp;than&nbsp;tail.</p>
</td>
</tr>
<tr>
<td width="512">
<p>&rsquo;dice&rsquo;- PMF:{1:1/6,2:1/6,3:1/6,4:1/6,5:1/6,6:1/6}.</p>
</td>
</tr>
<tr>
<td width="512">
<p>&rsquo;tampereddice&rsquo; - PMF: {1:0.1,2:0.1,3:0.1,4:0.1,5:0.1,6:0.5},i.e.,&nbsp;6&nbsp;is&nbsp;more&nbsp;likely.</p>
</td>
</tr>
<tr>
<td width="170">
<p>t_stat()</p>
</td>
<td width="512">
<p>This&nbsp;function&nbsp;constructs&nbsp;an&nbsp;r.v. &nbsp;(random&nbsp;variable) following&nbsp;the t&nbsp;distribution.</p>
</td>
</tr>
<tr>
<td width="170">
<p>chisq_gof_stat()</p>
</td>
<td width="512">
<p>Verify&nbsp;the&nbsp;statistic&nbsp;used&nbsp;in&nbsp;Pearson&rsquo;s&nbsp;Chi-Square&nbsp;Goodness-of-Fit test&nbsp;follows&nbsp;the&nbsp;&chi;2 &nbsp;distribution.</p>
</td>
</tr>
<tr>
<td width="170">
<p>fk_stat()</p>
</td>
<td width="512">
<p>Verify&nbsp;the&nbsp;Fligner-Killeen&nbsp;Test&nbsp;statistic(FK) follows&nbsp;the&nbsp;&chi;2 &nbsp;distribution.</p>
</td>
</tr>
<tr>
<td width="170">
<p>bartlett_stat()</p>
</td>
<td width="512">
<p>Verify&nbsp;the&nbsp;Bartlett&rsquo;s&nbsp;test&nbsp;statistic&nbsp;follows&nbsp;the&nbsp;&chi;2 &nbsp;distribution.</p>
</td>
</tr>
<tr>
<td width="170">
<p>anova_stat()</p>
</td>
<td width="512">
<p>Verify&nbsp;the&nbsp;statistic&nbsp;of&nbsp;ANOVA&nbsp;follows&nbsp;the&nbsp;F&nbsp;distribution.</p>
</td>
</tr>
<tr>
<td width="170">
<p>kw_stat()</p>
</td>
<td width="512">
<p>Verify&nbsp;the&nbsp;Kruskal-Wallis&nbsp;test&nbsp;statistic&nbsp;(H) is&nbsp;a&nbsp;&chi;2 &nbsp;r.v.</p>
</td>
</tr>
<tr>
<td width="170">
<p>sign_test_stat()</p>
</td>
<td width="512">
<p>For&nbsp;the&nbsp;sign&nbsp;test&nbsp;(medium&nbsp;test), verify&nbsp;its&nbsp;N- and&nbsp;N+ statistics&nbsp;both follow&nbsp;b(n,1/2).</p>
</td>
</tr>
<tr>
<td width="170">
<p>cochrane_q_stat()</p>
</td>
<td width="512">
<p>Verify&nbsp;the&nbsp;statistic&nbsp;T&nbsp;in&nbsp;Cochrane-Q&nbsp;test&nbsp;follows&nbsp;the&nbsp;&chi;2&nbsp;distribution.</p>
</td>
</tr>
<tr>
<td width="170">
<p>hotelling_t2_stat()</p>
</td>
<td width="512">
<p>Verify&nbsp;the&nbsp;T2 &nbsp;statistic&nbsp;from&nbsp;two&nbsp;multivariate&nbsp;Gaussian&nbsp;populations follows&nbsp;the&nbsp;Hotelling&rsquo;s&nbsp;T2 &nbsp;distribution.</p>
</td>
</tr>
</tbody>
</table>
</div>
# future plan

gui.py - add a Flask or tk-inter (ttkbootstrap) GUI
