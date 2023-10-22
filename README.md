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

<table border="1" cellspacing="0">
    <tbody>
        <tr>
            <td>
                Module
            </td>
            <td>
                Function
            </td>
            <td>
                Description
            </td>
        </tr>
        <tr>
            <td rowspan="7">
                mc.experiments
            </td>
            <td>
                pi()
            </td>
            <td>
                Perform Buffon&rsquo;s needle experiment to estimate&nbsp;&pi; .
            </td>
        </tr>
        <tr>
            <td>
                parcel()
            </td>
            <td>
                Simulate&nbsp;a&nbsp;bi-directional&nbsp;parcel&nbsp;passing&nbsp;game.
            </td>
        </tr>
        <tr>
            <td>
                dices()
            </td>
            <td>
                Estimate&nbsp;the&nbsp;probabilities&nbsp;of&nbsp;various&nbsp;dice&nbsp;combinations.
            </td>
        </tr>
        <tr>
            <td>
                prisoners()
                Prisoners_limit()
            </td>
            <td>
                The&nbsp;famous&nbsp;locker&nbsp;puzzle(100-prisoner&nbsp;quiz). This&nbsp;function&nbsp;will&nbsp;prove&nbsp;that&nbsp;the&nbsp;survival&nbsp;chance&nbsp;limit&nbsp;is&nbsp;1&minus;ln2 when&nbsp;n&nbsp;approaches&nbsp;+&infin; .
            </td>
        </tr>
        <tr>
            <td>
                galton_board()
            </td>
            <td>
                Use&nbsp;the&nbsp;classic&nbsp;Galton&nbsp;board&nbsp;experiment&nbsp;to&nbsp;produce&nbsp;a&nbsp;binomial&nbsp;distribution.
            </td>
        </tr>
        <tr>
            <td>
                paper_clips()
            </td>
            <td>
                Use&nbsp;the&nbsp;paper&nbsp;clip&nbsp;experiment&nbsp;to&nbsp;produce&nbsp;a&nbsp;Zipf&nbsp;distribution.
            </td>
        </tr>
        <tr>
            <td>
                sudden_death()
            </td>
            <td>
                This&nbsp;function&nbsp;simulates&nbsp;a&nbsp;sudden&nbsp;death&nbsp;game&nbsp;to&nbsp;produce the&nbsp;exponential&nbsp;distribution.
            </td>
        </tr>
        <tr>
            <td rowspan="2">
                mc.distributions
            </td>
            <td>
                poisson()
            </td>
            <td>
                This&nbsp;function&nbsp;will&nbsp;demonstrate&nbsp;that&nbsp;Poisson&nbsp;is&nbsp;a&nbsp;limit&nbsp;distribution&nbsp;of b(n,p) when&nbsp;n&nbsp;is&nbsp;large, and&nbsp;p&nbsp;is&nbsp;small.
            </td>
        </tr>
        <tr>
            <td>
                benford()
            </td>
            <td>
                Verify&nbsp;Benford&rsquo;s&nbsp;law&nbsp;using&nbsp;real-life&nbsp;datasets, including&nbsp;the&nbsp;stock market&nbsp;data, international&nbsp;trade&nbsp;data, and&nbsp;the&nbsp;Fibonacci&nbsp;series.
            </td>
        </tr>
        <tr>
            <td rowspan="17">
                mc.samplings
            </td>
            <td rowspan="8">
                clt()
            </td>
            <td>
                Using&nbsp;various&nbsp;underlying&nbsp;distributions&nbsp;to&nbsp;verify&nbsp;the&nbsp;central&nbsp;limit&nbsp;&nbsp;theorem.&nbsp;This&nbsp;function&nbsp;provides&nbsp;the&nbsp;following&nbsp;underlying&nbsp;distributions.
            </td>
        </tr>
        <tr>
            <td>
                &rsquo;uniform&rsquo; - a&nbsp;uniform&nbsp;distribution&nbsp;U(-1,1).
            </td>
        </tr>
        <tr>
            <td>
                &rsquo;expon&rsquo;- an&nbsp;exponential distribution Expon(1).
            </td>
        </tr>
        <tr>
            <td>
                &rsquo;poisson&rsquo; - poisson&nbsp;distribution &pi;(1).
            </td>
        </tr>
        <tr>
            <td>
                &rsquo;coin&rsquo;- Bernoulli&nbsp;distribution&nbsp;with&nbsp;p&nbsp;= 0.5.
            </td>
        </tr>
        <tr>
            <td>
                &rsquo;tampered_coin&rsquo;&nbsp;- PMF:{0:0.2,1:0.8}, i.e., head&nbsp;more&nbsp;likely&nbsp;than&nbsp;tail.
            </td>
        </tr>
        <tr>
            <td>
                &rsquo;dice&rsquo;- PMF:{1:1/6,2:1/6,3:1/6,4:1/6,5:1/6,6:1/6}.
            </td>
        </tr>
        <tr>
            <td>
                &rsquo;tampereddice&rsquo; - PMF: {1:0.1,2:0.1,3:0.1,4:0.1,5:0.1,6:0.5},i.e.,&nbsp;6&nbsp;is&nbsp;more&nbsp;likely.
            </td>
        </tr>
        <tr>
            <td>
                t_stat()
            </td>
            <td>
                This&nbsp;function&nbsp;constructs&nbsp;an&nbsp;r.v. &nbsp;(random&nbsp;variable) following&nbsp;the t&nbsp;distribution.
            </td>
        </tr>
        <tr>
            <td>
                chisq_gof_stat()
            </td>
            <td>
                Verify&nbsp;the&nbsp;statistic&nbsp;used&nbsp;in&nbsp;Pearson&rsquo;s&nbsp;Chi-Square&nbsp;Goodness-of-Fit test&nbsp;follows&nbsp;the&nbsp;&chi;2 &nbsp;distribution.
            </td>
        </tr>
        <tr>
            <td>
                fk_stat()
            </td>
            <td>
                Verify&nbsp;the&nbsp;Fligner-Killeen&nbsp;Test&nbsp;statistic(FK) follows&nbsp;the&nbsp;&chi;2 &nbsp;distribution.
            </td>
        </tr>
        <tr>
            <td>
                bartlett_stat()
            </td>
            <td>
                Verify&nbsp;the&nbsp;Bartlett&rsquo;s&nbsp;test&nbsp;statistic&nbsp;follows&nbsp;the&nbsp;&chi;2 &nbsp;distribution.
            </td>
        </tr>
        <tr>
            <td>
                anova_stat()
            </td>
            <td>
                Verify&nbsp;the&nbsp;statistic&nbsp;of&nbsp;ANOVA&nbsp;follows&nbsp;the&nbsp;F&nbsp;distribution.
            </td>
        </tr>
        <tr>
            <td>
                kw_stat()
            </td>
            <td>
                Verify&nbsp;the&nbsp;Kruskal-Wallis&nbsp;test&nbsp;statistic&nbsp;(H) is&nbsp;a&nbsp;&chi;2 &nbsp;r.v.
            </td>
        </tr>
        <tr>
            <td>
                sign_test_stat()
            </td>
            <td>
                For&nbsp;the&nbsp;sign&nbsp;test&nbsp;(medium&nbsp;test), verify&nbsp;its&nbsp;N- and&nbsp;N+ statistics&nbsp;both follow&nbsp;b(n,1/2).
            </td>
        </tr>
        <tr>
            <td>
                cochrane_q_stat()
            </td>
            <td>
                Verify&nbsp;the&nbsp;statistic&nbsp;T&nbsp;in&nbsp;Cochrane-Q&nbsp;test&nbsp;follows&nbsp;the&nbsp;&chi;2&nbsp;distribution.
            </td>
        </tr>
        <tr>
            <td>
                hotelling_t2_stat()
            </td>
            <td>
                Verify&nbsp;the&nbsp;T2 &nbsp;statistic&nbsp;from&nbsp;two&nbsp;multivariate&nbsp;Gaussian&nbsp;populations follows&nbsp;the&nbsp;Hotelling&rsquo;s&nbsp;T2 &nbsp;distribution.
            </td>
        </tr>
    </tbody>
</table>

# future plan

gui.py - add a Flask or tk-inter (ttkbootstrap) GUI
