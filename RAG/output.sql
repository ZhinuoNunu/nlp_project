 SELECT lang, COUNT(*) AS num_favorites FROM "050_ING" GROUP BY lang ORDER BY num_favorites DESC LIMIT 1;
 SELECT "author_name", "type"
     FROM "050_ING"
     WHERE "type" = 'original'
     ORDER BY LENGTH("author_name") DESC LIMIT 1;
 SELECT DISTINCT "author_name" FROM "050_ING" WHERE "retweets" = 0;
 SELECT * FROM "050_ING" WHERE links IS NULL;
 SELECT COUNT(DISTINCT author_name) FROM "050_ING";
 SELECT MAX(length(text)) AS longest_post FROM "050_ING";
 SELECT SUM(retweets) AS total_retweets FROM "050_ING";
 SELECT COUNT(*) FROM "050_ING" WHERE mention_ids IS NULL;
 SELECT author_name, SUM(retweets) AS total_retweets FROM "050_ING" GROUP BY author_name ORDER BY total_retweets DESC LIMIT 1;
 SELECT lang, SUM(favorites) AS total_favorites FROM "050_ING" GROUP BY lang ORDER BY total_favorites DESC LIMIT 1;
 SELECT "author_name", COUNT(*) AS word_count FROM "050_ING" GROUP BY "author_name" ORDER BY word_count DESC LIMIT 1;
 SELECT type, COUNT(*) AS frequency FROM "050_ING" GROUP BY type ORDER BY frequency DESC LIMIT 1;
 SELECT "author_name", SUM("retweets") AS total_retweets FROM "050_ING" GROUP BY "author_name" ORDER BY total_retweets DESC LIMIT 3;
 SELECT lang, SUM(favorites) AS total_favorites FROM "050_ING" GROUP BY lang ORDER BY total_favorites ASC LIMIT 5;
 SELECT a.author_name, COUNT(*) AS num_posts FROM "050_ING" a JOIN (SELECT MIN(length(text)) AS min_length FROM "050_ING") b ON 1=1 GROUP BY a.author_name HAVING COUNT(*) > 4;
 SELECT type, COUNT(*) AS frequency FROM "050_ING" GROUP BY type ORDER BY frequency DESC LIMIT 6;
 SELECT retweets, favorites FROM "050_ING" ORDER BY favorites DESC, retweets DESC LIMIT 5;
 SELECT LENGTH(text) AS word_count, text FROM "050_ING" ORDER BY word_count DESC LIMIT 3;
 SELECT retweets, favorites FROM "050_ING" ORDER BY favorites ASC LIMIT 4;
 SELECT LENGTH(text) AS word_count, text FROM "050_ING" ORDER BY word_count ASC LIMIT 6;
 SELECT name FROM "051_Pokemon" WHERE name = 'Pikachu';
 SELECT total FROM "051_Pokemon" WHERE total > 700;
 SELECT legendary FROM "051_Pokemon" WHERE generation = 1;
 SELECT EXISTS (SELECT 1 FROM "051_Pokemon" WHERE speed > 150);
 SELECT COUNT(DISTINCT type1) FROM "051_Pokemon";
 SELECT MAX(total) AS max_total FROM "051_Pokemon";
 SELECT COUNT(*) FROM "051_Pokemon" WHERE generation = 3;
 SELECT AVG(attack) AS average_attack FROM "051_Pokemon";
 SELECT t.type1, MAX(t.defense) AS max_defense FROM "051_Pokemon" t GROUP BY t.type1 ORDER BY max_defense DESC LIMIT 1;
 SELECT name, speed FROM "051_Pokemon" ORDER BY speed ASC LIMIT 1;
 SELECT t2.name AS secondary_type, COUNT(*) AS frequency FROM "051_Pokemon" p JOIN type2 t2 ON p.type2 = t2.id WHERE p.legendary = 'Yes' GROUP BY t2.name ORDER BY frequency DESC LIMIT 1;
 SELECT p.name, p.type1, p.sp_attack
     FROM "051_Pokemon" p
     ORDER BY p.sp_attack DESC NULLS LAST LIMIT 1;
 SELECT p.name, p.total
     FROM "051_Pokemon" p
     ORDER BY p.total DESC NULLS LAST
     LIMIT 3;
 SELECT name, hp FROM "051_Pokemon" ORDER BY hp ASC NULLS LAST LIMIT 5;
 SELECT t1.name, COUNT(*) AS num_pokemon FROM "051_Pokemon" p JOIN "052_Type" t1 ON p.type1 = t1.id GROUP BY t1.name ORDER BY num_pokemon DESC LIMIT 4;
 SELECT name, attack FROM "051_Pokemon" WHERE generation = 2 ORDER BY attack DESC NULLS LAST LIMIT 6;
 SELECT sp_defense FROM "051_Pokemon" ORDER BY sp_defense DESC NULLS LAST LIMIT 5;
 SELECT p.name, p.defense FROM "051_Pokemon" p WHERE p.legendary = 'Yes' ORDER BY p.defense ASC LIMIT 3;
 SELECT generation, MAX(speed) AS max_speed FROM "051_Pokemon" WHERE generation = 4 GROUP BY generation ORDER BY max_speed DESC LIMIT 4;
 SELECT * FROM (SELECT * FROM "051_Pokemon" WHERE legendary = 'No' ORDER BY total ASC LIMIT 6) AS subquery;
 SELECT MAX(Extraversion) > MAX(Agreeableness) AS comparison_result FROM "052_Professional";
 SELECT (SELECT Profession FROM "052_Professional" ORDER BY Openness DESC LIMIT 1) = (SELECT Profession FROM "052_Professional" ORDER BY Conscientousness DESC LIMIT 1) AS same_profession;
 SELECT p.Profession, p.Emotional_Range, p.Conversation FROM "052_Professional" p ORDER BY p.Emotional_Range ASC, p.Conversation ASC LIMIT 1;
 SELECT (AVG(Openness to Change)::FLOAT > AVG(Hedonism)::FLOAT) AS is_higher FROM "052_Professional";
 SELECT MAX(Self-enhancement) AS max_self_enhancement FROM "052_Professional";
 SELECT COUNT(*) FROM "052_Professional" WHERE Emotional_Range > 0.5;
 SELECT p.Profession, AVG(p.Extraversion) AS average_extraversion FROM "052_Professional" p GROUP BY p.Profession ORDER BY n DESC LIMIT 1;
 SELECT MIN(Self-transcendence) AS min_self_transcendence FROM "052_Professional";
 SELECT Profession, MAX(Conscientousness) AS max_conscientiousness FROM "052_Professional" GROUP BY Profession ORDER BY max_conscientiousness DESC LIMIT 1;
 SELECT Profession, MIN(Hedonism) AS min_hedonism FROM "052_Professional" GROUP BY Profession ORDER BY min_hedonism ASC LIMIT 1;
 SELECT Profession, MAX(Emotional_Range) AS max_emotional_range FROM "052_Professional" GROUP BY Profession ORDER BY max_emotional_range DESC NULLS LAST LIMIT 1;
 SELECT Profession, n FROM "052_Professional" ORDER BY n DESC LIMIT 1;
 SELECT p.Profession, p.Openness FROM "052_Professional" p ORDER BY p.Openness DESC NULLS LAST LIMIT 3;
 SELECT Profession, Agreeableness
     FROM "052_Professional"
     ORDER BY Agreeableness ASC
     LIMIT 4;
 SELECT p.Profession, p.Conversation FROM "052_Professional" p ORDER BY p.Conversation DESC NULLS LAST LIMIT 5;
 SELECT p.Profession, p.Self-enhancement FROM "052_Professional" p ORDER BY p.Self-enhancement ASC LIMIT 2;
 SELECT Profession, Openness to Change FROM "052_Professional" ORDER BY Openness to Change DESC NULLS LAST LIMIT 3;
 SELECT Emotional_Range FROM "052_Professional" ORDER BY Emotional_Range ASC LIMIT 4;
 SELECT "Extraversion" FROM "052_Professional" ORDER BY "Extraversion" DESC NULLS LAST LIMIT 5;
 SELECT Self-transcendence FROM "052_Professional" ORDER BY Self-transcendence ASC NULLS LAST LIMIT 6;
 SELECT title FROM "053_Patents" WHERE title ilike '%communication%';
 SELECT COUNT(*) FROM "053_Patents" WHERE organization = 'IBM';
 SELECT abstract FROM "053_Patents" WHERE abstract ilike '%software%';
 SELECT COUNT(*) FROM "053_Patents" WHERE type = 'design';
 SELECT COUNT(DISTINCT organization) FROM "053_Patents";
 SELECT AVG(num_claims) AS average_claims FROM "053_Patents";
 SELECT MAX(num_claims) AS max_num_claims FROM "053_Patents";
 SELECT COUNT(*) FROM "053_Patents" WHERE type = 'utility';
 SELECT organization, num_claims FROM "053_Patents" ORDER BY num_claims DESC LIMIT 1;
 SELECT kind, COUNT(*) AS frequency FROM "053_Patents" GROUP BY kind ORDER BY frequency DESC LIMIT 1;
 SELECT lang FROM "053_Patents";
 SELECT graphext_cluster, COUNT(*) AS frequency FROM "053_Patents" GROUP BY graphext_cluster ORDER BY frequency DESC LIMIT 1;
 SELECT organization, COUNT(*) AS num_patents FROM "053_Patents" GROUP BY organization ORDER BY num_patents DESC LIMIT 3;
 SELECT type, COUNT(*) AS frequency FROM "053_Patents" GROUP BY type ORDER BY frequency DESC LIMIT 4;
 SELECT kind, COUNT(*) AS frequency FROM "053_Patents" GROUP BY kind ORDER BY frequency DESC LIMIT 5;
 SELECT graphext_cluster, COUNT(*) AS frequency FROM "053_Patents" GROUP BY graphext_cluster ORDER BY frequency ASC LIMIT 2;
 SELECT num_claims FROM "053_Patents" ORDER BY num_claims DESC NULLS LAST LIMIT 4;
 SELECT p.id, p.num_claims FROM "053_Patents" p ORDER BY p.num_claims DESC NULLS LAST LIMIT 3;
 SELECT p.id, COUNT(p.target) AS target_count FROM "053_Patents" p GROUP BY p.id ORDER BY target_count DESC LIMIT 5;
 SELECT id, date FROM "053_Patents" ORDER BY date DESC LIMIT 6;
 SELECT "author_id<gx:category>", MAX("user_followers_count<gx:number>") AS max_followers, MIN("user_verified<gx:boolean>") AS min_verified FROM "054_Joe" GROUP BY "author_id<gx:category>" HAVING MAX("user_followers_count<gx:number>") > 1000 AND MIN("user_verified<gx:boolean>") = TRUE;
 SELECT "author_id<gx:category>", MAX("user_favourites_count<gx:number>") AS max_favourites, MAX("retweets<gx:number>") AS max_retweets FROM "054_Joe" GROUP BY "author_id<gx:category>" ORDER BY max_favourites DESC, max_retweets DESC LIMIT 1;
 SELECT "author_id<gx:category>", COUNT("mention_names<gx:list[category]>") AS num_mentions, SUM("retweets<gx:number>") AS total_retweets
     FROM "054_Joe"
     GROUP BY "author_id<gx:category>"
     ORDER BY num_mentions DESC, total_retweets DESC
     LIMIT 1;
 SELECT "author_id<gx:category>", MAX("retweets<gx:number>") AS max_retweets, MAX("replies<gx:number>") AS max_replies FROM "054_Joe" GROUP BY "author_id<gx:category>" ORDER BY max_retweets DESC, max_replies DESC LIMIT 1;
 SELECT MAX(user_followers_count) AS max_followers FROM "054_Joe";
 SELECT COUNT(*) FROM "054_Joe" WHERE user_favourites_count > 10000;
 SELECT MAX(retweets) AS max_retweets FROM "054_Joe";
 SELECT COUNT(*) AS num_mentions FROM "054_Joe" WHERE mention_names IS NOT NULL;
 SELECT "author_name<gx:category>", MAX("user_followers_count<gx:number>") AS max_followers FROM "054_Joe" GROUP BY "author_name<gx:category>" ORDER BY max_followers DESC LIMIT 1;
 SELECT "author_name<gx:category>", MAX("user_favourites_count<gx:number>") AS max_favourites FROM "054_Joe" GROUP BY "author_name<gx:category>" ORDER BY max_favourites DESC LIMIT 1;
 SELECT "author_name<gx:category>" FROM "054_Joe" GROUP BY "author_name<gx:category>" ORDER BY COUNT("mention_names<gx:list[category]>") DESC LIMIT 1;
 SELECT "author_name<gx:category>", MAX("retweets<gx:number>") AS max_retweets FROM "054_Joe" GROUP BY "author_name<gx:category>" ORDER BY max_retweets DESC LIMIT 1;
 SELECT "author_name<gx:category>", SUM("user_followers_count<gx:number>") AS total_followers FROM "054_Joe" GROUP BY "author_name<gx:category>" ORDER BY total_followers DESC LIMIT 3;
 SELECT "author_name<gx:category>", SUM("user_favourites_count<gx:number>") AS total_favourites FROM "054_Joe" GROUP BY "author_name<gx:category>" ORDER BY total_favourites DESC LIMIT 4;
 SELECT "author_name<gx:category>", COUNT(*) AS num_mentions FROM "054_Joe" GROUP BY "author_name<gx:category>" ORDER BY num_mentions DESC LIMIT 5;
 SELECT "author_name<gx:category>", SUM("retweets<gx:number>") AS total_retweets FROM "054_Joe" GROUP BY "author_name<gx:category>" ORDER BY total_retweets DESC LIMIT 2;
 SELECT user_followers_count<gx:number> AS followers_count, rank() OVER (ORDER BY user_followers_count<gx:number> DESC) AS rank FROM "054_Joe" ORDER BY rank ASC LIMIT 3;
 SELECT user_favourites_count FROM "054_Joe" ORDER BY user_favourites_count DESC NULLS LAST LIMIT 4;
 SELECT mention_names, COUNT(*) AS num_mentions FROM "054_Joe" GROUP BY mention_names ORDER BY num_mentions DESC LIMIT 5;
 SELECT MAX(retweets) AS max_retweets, COUNT(*) AS num_tweets FROM "054_Joe" GROUP BY max_retweets ORDER BY max_retweets DESC LIMIT 2;
 SELECT "Loan Amount", "Loan Duration - Months"
     FROM "055_German"
     ORDER BY "Loan Amount" DESC, "Loan Duration - Months" ASC LIMIT 1;
 SELECT MAX(Number of Existing Loans) AS max_loans, MAX(Loan Amount) AS max_loan_amount FROM "055_German";
 SELECT "Loan Duration - Months", "Number of Existing Loans"
     FROM "055_German"
     ORDER BY "Loan Duration - Months" DESC, "Number of Existing Loans" DESC LIMIT 1;
 SELECT Age, Loan Amount
     FROM "055_German"
     ORDER BY Age ASC, Loan Amount DESC
     LIMIT 1;
 SELECT MAX(Loan Amount) AS max_loan_amount FROM "055_German";
 SELECT COUNT(*) FROM "055_German" WHERE "Number of Existing Loans" > 1;
 SELECT MAX('Loan Duration - Months') AS longest_loan_duration FROM "055_German";
 SELECT COUNT(*) FROM "055_German" WHERE age > 50;
 SELECT "Purpose of Loan", COUNT(*) AS frequency FROM "055_German" GROUP BY "Purpose of Loan" ORDER BY frequency DESC LIMIT 1;
 SELECT Job, COUNT(*) AS frequency FROM "055_German" GROUP BY Job ORDER BY frequency DESC LIMIT 1;
 SELECT Credit History, COUNT(*) AS frequency FROM "055_German" GROUP BY Credit History ORDER BY frequency DESC LIMIT 1;
 SELECT Savings Account, COUNT(*) AS frequency FROM "055_German" GROUP BY Savings Account ORDER BY frequency DESC LIMIT 1;
 SELECT Job, SUM(Loan Amount) AS total_loan_amount FROM "055_German" GROUP BY Job ORDER BY total_loan_amount DESC LIMIT 3;
 SELECT Job, MAX(Loan Duration - Months) AS max_duration FROM "055_German" GROUP BY Job ORDER BY max_duration DESC LIMIT 4;
 SELECT j.job, COUNT(*) AS num_loans FROM "055_German" g JOIN "055_German" j ON g.borrower = j.borrower GROUP BY j.job ORDER BY num_loans DESC LIMIT 5;
 SELECT "Job" FROM "055_German" ORDER BY "Age" ASC LIMIT 2;
 SELECT Loan Amount FROM "055_German" ORDER BY Loan Amount DESC NULLS LAST LIMIT 3;
 SELECT Loan Duration - Months FROM 055_German ORDER BY Loan Duration - Months DESC NULLS LAST LIMIT 4;
 SELECT MAX(Number of Existing Loans) AS max_loans FROM "055_German";
 SELECT TOP 2 age FROM "055_German" ORDER BY age DESC NULLS LAST;
 SELECT "Calories (kcal)" FROM "056_Emoji" WHERE "Calories (kcal)" = 0;
 SELECT 'Total Sugar (g)' FROM '056_Emoji';
 SELECT "Food" FROM "056_Emoji" WHERE "Total Fat (g)" = 0;
 SELECT "Food" FROM "056_Emoji" WHERE "Protein (g)" IS NULL;
 SELECT COUNT(*) FROM "056_Emoji" WHERE "Calories (kcal)" > 500;
 SELECT AVG(Total Fat (g)) AS average_total_fat FROM "056_Emoji";
 SELECT COUNT(*) FROM "056_Emoji" WHERE "Sodium (g)" > 1;
 SELECT MAX("Protein (g)") FROM "056_Emoji";
 SELECT name, calories_kcal FROM "056_Emoji" ORDER BY calories_kcal DESC LIMIT 1;
 SELECT name, SUM(Total Sugar (g)) AS total_sugar FROM "056_Emoji" GROUP BY name ORDER BY total_sugar DESC LIMIT 1;
 SELECT name, SUM(Total Fat (g)) AS total_fat FROM "056_Emoji" GROUP BY name ORDER BY total_fat ASC LIMIT 1;
 SELECT name, protein_g FROM "056_Emoji" ORDER BY protein_g DESC NULLS LAST LIMIT 1;
 SELECT name, calories_kcal FROM 056_emoji ORDER BY calories_kcal DESC NULLS LAST LIMIT 5;
 SELECT name, SUM(Total Sugar (g)) AS total_sugar FROM "056_Emoji" GROUP BY name ORDER BY total_sugar ASC LIMIT 3;
 SELECT name, SUM(Total Fat (g)) AS total_fat FROM "056_Emoji" GROUP BY name ORDER BY total_fat DESC LIMIT 4;
 SELECT name, protein_g FROM "056_Emoji" ORDER BY protein_g ASC LIMIT 6;
 SELECT MAX(Calories (kcal)) AS max_calories FROM "056_Emoji";
 SELECT MIN(Total Sugar (g)) AS min_sugar FROM "056_Emoji";
 SELECT Total Fat (g) FROM "056_Emoji" ORDER BY Total Fat (g) DESC NULLS LAST LIMIT 4;
 SELECT Protein (g) FROM "056_Emoji" ORDER BY Protein (g) ASC LIMIT 6;
 SELECT 'edu' FROM '057_Spain' WHERE 'edu' = 'No';
 SELECT polInterest FROM "057_Spain"
 SELECT "Vote Intention" FROM "057_Spain" WHERE "Vote Intention" = 'No';
 SELECT COUNT(*) FROM "057_Spain" WHERE "User self-placement on Left-Right economic values axis" = 'Right';
 SELECT COUNT(*) FROM "057_Spain" WHERE "User self- placement on Progressive-Conservative economic values axis" = 10;
 SELECT AVG(Age) AS average_age FROM "057_Spain";
 SELECT COUNT(*) FROM "057_Spain" WHERE "It should be more difficult for companies to lay off workers" = 'Yes';
 SELECT MAX(age) FROM "057_Spain" WHERE gender IS NULL;
 SELECT Vote Intention, COUNT(*) AS frequency FROM "057_Spain" GROUP BY Vote Intention ORDER BY frequency DESC LIMIT 1;
 SELECT "voteReason", COUNT(*) AS frequency FROM "057_Spain" GROUP BY "voteReason" ORDER BY frequency DESC LIMIT 1;
 SELECT partyId, COUNT(*) AS frequency FROM "057_Spain" GROUP BY partyId ORDER BY frequency DESC LIMIT 1;
 SELECT lang FROM "057_Spain" GROUP BY lang ORDER BY COUNT(lang) DESC LIMIT 1;
 SELECT Vote Intention, COUNT(*) AS frequency FROM "057_Spain" GROUP BY Vote Intention ORDER BY frequency DESC LIMIT 5;
 SELECT voteReason, COUNT(*) AS frequency FROM "057_Spain" GROUP BY voteReason ORDER BY frequency DESC LIMIT 3;
 SELECT partyId, COUNT(*) AS frequency FROM "057_Spain" GROUP BY partyId ORDER BY frequency DESC LIMIT 4;
 SELECT lang, COUNT(*) AS frequency FROM "057_Spain" GROUP BY lang ORDER BY frequency DESC LIMIT 6;
 SELECT age, COUNT(*) AS frequency FROM "057_Spain" GROUP BY age ORDER BY frequency DESC LIMIT 5;
 SELECT "User self-placement on Left-Right economic values axis", COUNT(*) AS frequency FROM "057_Spain" GROUP BY "User self-placement on Left-Right economic values axis" ORDER BY frequency DESC LIMIT 3;
 SELECT "User self- placement on Progressive-Conservative economic values axis", COUNT(*) AS frequency FROM "057_Spain" GROUP BY "User self- placement on Progressive-Conservative economic values axis" ORDER BY frequency DESC LIMIT 4;
 SELECT "Constitutional organisation of the country (More Descentralization)", COUNT(*) AS frequency FROM "057_Spain" GROUP BY "Constitutional organisation of the country (More Descentralization)" ORDER BY frequency DESC LIMIT 6;
 SELECT "Are you registered to vote?" FROM "058_US" WHERE "Are you registered to vote?" = 'No';
 SELECT "Did you vote in the 2016 Presidential election? (Four years ago)", "How likely are you to vote in the forthcoming US Presidential election? Early Voting Open"
     FROM "058_US"
     WHERE "Did you vote in the 2016 Presidential election? (Four years ago)" = 'Yes' AND "How likely are you to vote in the forthcoming US Presidential election? Early Voting Open" NOT ILIKE '%Yes%';
 SELECT "Did you vote in the 2016 Presidential election? (Four years ago)", "How likely are you to vote in the forthcoming US Presidential election? Early Voting Open"
     FROM "058_US"
     WHERE "Did you vote in the 2016 Presidential election? (Four years ago)" = 'No' AND "How likely are you to vote in the forthcoming US Presidential election? Early Voting Open" BETWEEN 1 AND 3;
 SELECT COUNT(*) FROM "058_US" WHERE "Who did you vote for in the 2016 Presidential election? (Four years ago)"!= "Who are you most likely to vote for on election day?"
 SELECT COUNT(*) FROM "058_US" WHERE "Are you registered to vote?" = 'Yes';
 SELECT COUNT(*) FROM "058_US" WHERE "Who are you most likely to vote for on election day?" = 'Joe Biden';
 SELECT COUNT(*) FROM "058_US" WHERE 'What is the highest degree or level of school you have *completed*?' IN ('High school', 'High school or equivalent', 'Some college or equivalent', 'Less than high school');
 SELECT COUNT(*) FROM "058_US" WHERE Division = 'South Atlantic';
 SELECT CAST(HOW_OLD_ARE_YOU AS INTEGER) AS age_group, COUNT(*) AS frequency FROM "058_US" GROUP BY age_group ORDER BY frequency DESC LIMIT 1;
 SELECT "Who are you most likely to vote for on election day?" AS preferred_candidate FROM "058_US";
 SELECT "What is the highest degree or level of school you have *completed*?", COUNT(*) AS frequency FROM "058_US" GROUP BY "What is the highest degree or level of school you have *completed*?" ORDER BY frequency DESC LIMIT 1;
 SELECT Division, COUNT(*) AS representation FROM "058_US" GROUP BY Division ORDER BY representation DESC LIMIT 1;
 SELECT State, COUNT(*) AS num_respondents FROM "058_US" GROUP BY State ORDER BY num_respondents DESC LIMIT 5;
 SELECT "Which of these best describes the kind of work you do?", COUNT(*) AS frequency FROM "058_US" GROUP BY "Which of these best describes the kind of work you do?" ORDER BY frequency DESC LIMIT 4;
 SELECT "Which of the following best describes your ethnic heritage?", COUNT(*) AS frequency FROM "058_US" GROUP BY "Which of the following best describes your ethnic heritage?" ORDER BY frequency DESC LIMIT 3;
 SELECT CAST(HOW_OLD_ARE_YOU AS INTEGER) AS age_group, COUNT(*) AS num_respondents FROM "058_US" GROUP BY age_group ORDER BY num_respondents DESC LIMIT 6;
 SELECT "County FIPS", COUNT(*) AS num_respondents FROM "058_US" GROUP BY "County FIPS" ORDER BY num_respondents DESC LIMIT 3;
 SELECT MAX(Weight) AS max_weight FROM "058_US" GROUP BY Weight ORDER BY max_weight DESC NULLS LAST LIMIT 4;
 SELECT "NCHS Urban/rural", COUNT(*) AS num_respondents FROM "058_US" GROUP BY "NCHS Urban/rural" ORDER BY num_respondents DESC LIMIT 5;
 SELECT likelihood, COUNT(*) AS frequency FROM "058_US" GROUP BY likelihood ORDER BY frequency DESC LIMIT 6;
 SELECT model FROM "059_Second" WHERE model ilike '%Golf%';
 SELECT COUNT(*) FROM "059_Second" s WHERE s.dealer = 'Autos Raymara';
 SELECT version FROM "059_Second" WHERE version ilike '%BMW%';
 SELECT'shift' FROM "059_Second" WHERE'shift' = 'Manual';
 SELECT COUNT(DISTINCT model) FROM "059_Second";
 SELECT AVG(photos) AS average_photos FROM "059_Second";
 SELECT MAX(price) AS max_price FROM "059_Second";
 SELECT COUNT(*) FROM "059_Second" WHERE year = 2020;
 SELECT dealer, MAX(price) AS max_price FROM "059_Second" GROUP BY dealer;
 SELECT make, COUNT(*) AS COUNT FROM "059_Second" GROUP BY make ORDER BY COUNT DESC LIMIT 1;
 SELECT fuel, COUNT(*) AS COUNT FROM "059_Second" GROUP BY fuel ORDER BY COUNT DESC LIMIT 1;
 SELECT province, COUNT(*) AS num_cars FROM "059_Second" GROUP BY province ORDER BY num_cars DESC LIMIT 1;
 SELECT province, COUNT(*) AS num_listings FROM "059_Second" GROUP BY province ORDER BY num_listings DESC LIMIT 3;
 SELECT color, COUNT(*) AS frequency FROM "059_Second" GROUP BY color ORDER BY frequency DESC LIMIT 4;
 SELECT make, COUNT(*) AS frequency FROM "059_Second" GROUP BY make ORDER BY frequency DESC LIMIT 5;
 SELECT fuel, COUNT(*) AS frequency FROM "059_Second" GROUP BY fuel ORDER BY frequency ASC LIMIT 2;
 SELECT year, COUNT(*) AS frequency FROM "059_Second" GROUP BY year ORDER BY frequency DESC LIMIT 4;
 SELECT kms FROM "059_Second" ORDER BY kms DESC NULLS LAST LIMIT 3;
 SELECT c.name, COUNT(p.id) AS photo_count FROM "059_Second" c JOIN "059_Photos" p ON c.id = p.car_id GROUP BY c.name ORDER BY photo_count DESC LIMIT 5;
 SELECT * FROM "059_Second" ORDER BY power DESC LIMIT 6;
 SELECT COUNT(DISTINCT Transaction) AS num_unique_transactions FROM "060_Bakery";
 SELECT COUNT(*) FROM "060_Bakery" WHERE period_day = 'Night';
 SELECT "Item", COUNT(*) AS num_transactions FROM "060_Bakery" WHERE "weekday_weekend" = 'Weekday' GROUP BY "Item" HAVING COUNT(*) > 0;
 SELECT period_day, weekday_weekend FROM "060_Bakery" WHERE period_day = 'Evening' AND weekday_weekend = 'Yes';
 SELECT COUNT(DISTINCT Item) FROM "060_Bakery";
 SELECT COUNT(DISTINCT date_time) AS num_days FROM "060_Bakery";
 SELECT MAX(Transaction) AS max_transaction FROM "060_Bakery";
 SELECT COUNT(*) FROM "060_Bakery" WHERE period_day = 'Afternoon';
 SELECT "period_day", SUM("Transaction") AS total_transactions FROM "060_Bakery" GROUP BY "period_day" ORDER BY total_transactions DESC LIMIT 1;
 SELECT "Item", COUNT(*) AS num_sales FROM "060_Bakery" WHERE "weekday_weekend" = 'Weekend' GROUP BY "Item" ORDER BY num_sales DESC LIMIT 1;
 SELECT i.item, COUNT(*) AS num_purchases FROM "060_Bakery" i WHERE i.weekday_weekend = 'Weekday' GROUP BY i.item ORDER BY num_purchases ASC LIMIT 1;
 SELECT period_day, COUNT(*) AS frequency FROM "060_Bakery" WHERE item = 'Hot chocolate' GROUP BY period_day ORDER BY frequency DESC LIMIT 1;
 SELECT "Item", COUNT(*) AS num_purchases FROM "060_Bakery" WHERE "period_day" = 'Morning' GROUP BY "Item" ORDER BY num_purchases DESC LIMIT 3;
 SELECT Item, COUNT(*) AS num_purchases FROM "060_Bakery" WHERE period_day = 'Afternoon' GROUP BY Item ORDER BY num_purchases ASC LIMIT 4;
 SELECT i.Item, SUM(i.Sales) AS total_sales FROM "060_Bakery" i WHERE i.weekday_weekend = 'Weekend' GROUP BY i.Item ORDER BY total_sales DESC LIMIT 5;
 SELECT Item FROM "060_Bakery" WHERE period_day = 'Evening' GROUP BY Item HAVING COUNT(*) >= 2 LIMIT 4;
 SELECT "Transaction", COUNT(*) AS num_items_bought FROM "060_Bakery" GROUP BY "Transaction" ORDER BY num_items_bought DESC LIMIT 4;
 SELECT t.Transaction, COUNT(*) AS frequency FROM "060_Bakery" t GROUP BY t.Transaction ORDER BY frequency ASC LIMIT 5;
 SELECT "Transaction", COUNT(*) AS num_transactions FROM "060_Bakery" WHERE "Item" = 'Bread' GROUP BY "Transaction" ORDER BY num_transactions DESC LIMIT 6;
 SELECT t.transaction, t.period_day FROM "060_Bakery" t WHERE t.period_day = 'Morning' ORDER BY t.transaction ASC LIMIT 2;
 SELECT 'Reviewer_Location', 'Rating' FROM '061_Disneyland' WHERE 'Reviewer_Location' = 'Australia' AND 'Rating' > 3;
 SELECT Branch FROM "061_Disneyland" WHERE Branch = 'Disneyland_HongKong'
 SELECT Rating FROM "061_Disneyland" WHERE Rating = 1;
 SELECT rl.Reviewer_Location, COUNT(DISTINCT r.Review_ID) AS num_reviews FROM "061_Disneyland" r JOIN rl.Reviewer_Location rl ON r.Reviewer_Location = rl.Reviewer_Location WHERE r.Rating = 5 GROUP BY rl.Reviewer_Location HAVING COUNT(DISTINCT r.Review_ID) > 0;
 SELECT COUNT(DISTINCT Reviewer_Location) FROM "061_Disneyland";
 SELECT AVG(Rating) AS average_rating FROM "061_Disneyland" WHERE Branch = 'Disneyland_HongKong';
 SELECT MAX(Review_ID) FROM "061_Disneyland";
 SELECT COUNT(*) FROM "061_Disneyland" WHERE "Year_Month" LIKE '2019%';
 SELECT Reviewer_Location, COUNT(*) AS review_count FROM "061_Disneyland" GROUP BY Reviewer_Location ORDER BY review_count DESC LIMIT 1;
 SELECT Branch, AVG(Rating) AS avg_rating FROM "061_Disneyland" GROUP BY Branch ORDER BY avg_rating ASC LIMIT 1;
 SELECT "Year_Month", MIN(Rating) AS min_rating FROM "061_Disneyland" WHERE Rating = 1 GROUP BY "Year_Month" ORDER BY "Year_Month" NULLS LAST;
 SELECT Reviewer_Location, AVG(Rating) AS average_rating FROM "061_Disneyland" GROUP BY Reviewer_Location ORDER BY average_rating DESC LIMIT 1;
 SELECT "Reviewer_Location", COUNT(*) AS review_count FROM "061_Disneyland" GROUP BY "Reviewer_Location" ORDER BY review_count DESC LIMIT 3;
 SELECT Branch, AVG(Rating) AS avg_rating FROM "061_Disneyland" GROUP BY Branch ORDER BY avg_rating ASC LIMIT 2;
 SELECT "Year_Month", COUNT(*) AS review_count FROM "061_Disneyland" GROUP BY "Year_Month" ORDER BY review_count DESC LIMIT 4;
 SELECT rl.Reviewer_Location, AVG(rl.Rating) AS avg_rating FROM "061_Disneyland" rl GROUP BY rl.Reviewer_Location ORDER BY avg_rating ASC LIMIT 3;
 SELECT Review_ID, Rating FROM "061_Disneyland" ORDER BY Rating DESC NULLS LAST LIMIT 5;
 SELECT Review_ID, Rating
     FROM "061_Disneyland"
     ORDER BY Rating ASC
     LIMIT 4;
 SELECT "Reviewer_Location", "Rating" FROM "061_Disneyland" WHERE "Reviewer_Location" = 'Australia' ORDER BY "Rating" DESC NULLS LAST LIMIT 6;
 SELECT Branch, Rating
     FROM "061_Disneyland"
     WHERE Branch = 'Disneyland_HongKong'
     ORDER BY Rating ASC
     LIMIT 3;
 SELECT lang FROM "062_Trump" WHERE lang = 'en';
 SELECT COUNT(*) > 0 AS has_been_retweeted FROM "062_Trump" WHERE "retweets" IS NOT NULL;
 SELECT COUNT(*) FROM "062_Trump" WHERE retweets > 10000;
 SELECT favorites FROM "062_Trump" WHERE favorites > 50000;
 SELECT COUNT(DISTINCT author_name) FROM "062_Trump";
 SELECT AVG(retweets) AS average_retweets FROM "062_Trump";
 SELECT MAX(favorites) AS max_favorites FROM "062_Trump";
 SELECT COUNT(*) FROM "062_Trump" WHERE date BETWEEN '2018-01-01' AND '2018-12-31';
 SELECT author_name FROM "062_Trump" GROUP BY author_name ORDER BY COUNT(*) DESC LIMIT 1;
 SELECT text, favorites FROM "062_Trump" ORDER BY favorites DESC LIMIT 1;
 SELECT lang, SUM(retweets) AS total_retweets FROM "062_Trump" GROUP BY lang ORDER BY total_retweets DESC LIMIT 1;
 SELECT lang, COUNT(*) AS frequency FROM "062_Trump" GROUP BY lang ORDER BY frequency DESC LIMIT 1;
 SELECT author_handler, COUNT(*) AS num_tweets FROM "062_Trump" GROUP BY author_handler ORDER BY num_tweets DESC LIMIT 3;
 SELECT lang, COUNT(*) AS tweet_count FROM "062_Trump" GROUP BY lang ORDER BY tweet_count ASC LIMIT 2;
 SELECT mention_names FROM "062_Trump" GROUP BY mention_names ORDER BY COUNT(mention_names) DESC LIMIT 4;
 SELECT author_name, COUNT(*) AS tweet_count FROM 062_Trump GROUP BY author_name ORDER BY tweet_count ASC LIMIT 3;
 SELECT id, retweets FROM 062_Trump ORDER BY retweets DESC LIMIT 5;
 SELECT id, favorites FROM "062_Trump" ORDER BY favorites ASC LIMIT 4;
 SELECT favorites FROM "062_Trump" ORDER BY favorites DESC NULLS LAST LIMIT 6;
 SELECT tweet, COUNT(*) AS retweet_count FROM "062_Trump" GROUP BY tweet ORDER BY retweet_count ASC LIMIT 3;
 SELECT COUNT(*) FROM "063_Influencers" WHERE "is_organization" = 'Yes';
 SELECT COUNT(*) FROM "063_Influencers" WHERE is_organization = 'No';
 SELECT "pic" FROM "063_Influencers";
 SELECT * FROM "063_Influencers" WHERE weight > 500;
 SELECT COUNT(DISTINCT community) FROM "063_Influencers";
 SELECT AVG(page_rank_norm) AS average_page_rank_norm FROM "063_Influencers";
 SELECT MAX(weight) FROM "063_Influencers";
 SELECT COUNT(*) FROM "063_Influencers" WHERE community = 16744206;
 SELECT name FROM "063_Influencers" GROUP BY name ORDER BY COUNT(*) DESC LIMIT 1;
 SELECT name, page_rank_norm FROM "063_Influencers" ORDER BY page_rank_norm DESC NULLS LAST LIMIT 1;
 SELECT pic FROM "063_Influencers" WHERE weight = (SELECT MAX(weight) FROM "063_Influencers");
 SELECT name, y FROM "063_Influencers" ORDER BY y DESC NULLS LAST LIMIT 1;
 SELECT name, weight FROM "063_Influencers" ORDER BY weight DESC NULLS LAST LIMIT 3;
 SELECT name, page_rank_norm FROM "063_Influencers" ORDER BY page_rank_norm ASC LIMIT 2;
 SELECT name, x FROM "063_Influencers" ORDER BY x DESC NULLS LAST LIMIT 4;
 SELECT name, y
     FROM "063_Influencers"
     ORDER BY y ASC
     LIMIT 3;
 SELECT id, weight FROM "063_Influencers" ORDER BY weight DESC NULLS LAST LIMIT 5;
 SELECT id, page_rank_norm FROM "063_Influencers" ORDER BY page_rank_norm ASC LIMIT 4;
 SELECT page_rank_norm FROM "063_Influencers" ORDER BY page_rank_norm DESC NULLS LAST LIMIT 6;
 SELECT weight FROM "063_Influencers" ORDER BY weight ASC LIMIT 3;
 SELECT 'feathers' FROM "064_Clustering" WHERE 'feathers' IS NOT NULL;
 SELECT venomous FROM "064_Clustering" WHERE venomous = 'Yes';
 SELECT breathes FROM "064_Clustering" WHERE breathes = 'Yes';
 SELECT domestic FROM "064_Clustering" WHERE domestic = 'Yes';
 SELECT COUNT(DISTINCT class_type) FROM "064_Clustering";
 SELECT AVG(legs) AS average_legs FROM "064_Clustering";
 SELECT MAX(legs) FROM "064_Clustering";
 SELECT COUNT(*) FROM "064_Clustering" WHERE legs = 2;
 SELECT class_type, COUNT(*) AS frequency FROM "064_Clustering" GROUP BY class_type ORDER BY frequency DESC LIMIT 1;
 SELECT animal_name FROM "064_Clustering" WHERE legs = 8;
 SELECT c.class_type, MAX(c.legs) AS max_legs FROM "064_Clustering" c GROUP BY c.class_type ORDER BY max_legs DESC LIMIT 1;
 SELECT animal_name FROM "064_Clustering" WHERE venomous = 'Yes';
 SELECT animal_name, legs FROM "064_Clustering" ORDER BY legs DESC NULLS LAST LIMIT 3;
 SELECT animal_name, legs FROM "064_Clustering" ORDER BY legs ASC LIMIT 2;
 SELECT class_type, COUNT(*) AS num_animals FROM "064_Clustering" GROUP BY class_type ORDER BY num_animals DESC LIMIT 4;
 SELECT class_type, COUNT(*) AS num_animals FROM "064_Clustering" GROUP BY class_type ORDER BY num_animals ASC LIMIT 3;
 SELECT c.class_type, SUM(c.legs) AS total_legs FROM "064_Clustering" c GROUP BY c.class_type ORDER BY total_legs DESC LIMIT 5;
 SELECT c.class_type, SUM(c.legs) AS total_legs FROM "064_Clustering" c GROUP BY c.class_type ORDER BY total_legs ASC LIMIT 4;
 SELECT legs, COUNT(*) AS frequency FROM "064_Clustering" GROUP BY legs ORDER BY frequency DESC LIMIT 6;
 SELECT legs, COUNT(*) AS frequency FROM "064_Clustering" GROUP BY legs ORDER BY frequency ASC LIMIT 3;
 SELECT Quantity > 0 AS all_quantities_greater_than_zero FROM "065_RFM";
 SELECT UnitPrice FROM "065_RFM" WHERE UnitPrice > 0;
 SELECT "CustomerID" FROM "065_RFM" WHERE "CustomerID" IS NULL;
 SELECT COUNT(*) FROM "065_RFM" WHERE Country = 'United Kingdom';
 SELECT COUNT(DISTINCT StockCode) FROM "065_RFM";
 SELECT AVG(UnitPrice) AS average_unit_price FROM "065_RFM";
 SELECT MAX(Quantity) AS max_quantity FROM "065_RFM";
 SELECT COUNT(*) FROM "065_RFM" WHERE Country = 'United Kingdom';
 SELECT Country, COUNT(*) AS num_transactions FROM "065_RFM" GROUP BY Country ORDER BY num_transactions DESC LIMIT 1;
 SELECT Description, UnitPrice FROM "065_RFM" ORDER BY UnitPrice DESC NULLS LAST LIMIT 1;
 SELECT Description, Quantity
     FROM "065_RFM"
     ORDER BY Quantity DESC
     LIMIT 1;
 SELECT Country, MIN(CustomerID) AS min_customer_id FROM "065_RFM" GROUP BY Country ORDER BY min_customer_id NULLS LAST;
 SELECT Description, Quantity FROM "065_RFM" ORDER BY Quantity DESC LIMIT 3;
 SELECT Description, Quantity
     FROM "065_RFM"
     ORDER BY Quantity ASC
     LIMIT 2;
 SELECT Country, COUNT(*) AS num_transactions FROM "065_RFM" GROUP BY Country ORDER BY num_transactions DESC LIMIT 4;
 SELECT Country, COUNT(*) AS num_transactions FROM "065_RFM" GROUP BY Country ORDER BY num_transactions ASC LIMIT 3;
 SELECT i.InvoiceNo, i.Quantity FROM 065_RFM i ORDER BY i.Quantity DESC NULLS LAST LIMIT 5;
 SELECT i.InvoiceNo, i.Quantity
     FROM "065_RFM" i
     ORDER BY i.Quantity ASC
     LIMIT 4;
 SELECT Quantity, COUNT(*) AS frequency FROM "065_RFM" GROUP BY Quantity ORDER BY frequency DESC LIMIT 6;
 SELECT Quantity FROM "065_RFM" GROUP BY Quantity ORDER BY COUNT(Quantity) ASC LIMIT 3;
