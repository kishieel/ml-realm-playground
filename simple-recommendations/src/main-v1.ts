import * as tf from '@tensorflow/tfjs'

const users = ['Adam', 'Bartek', 'Czesiek', 'Daniel'];
const bands = ['Nirvana', 'Nine Inch Nails', 'Backstreet Boys', 'N Sync', 'Night Club', 'Apashe', 'STP'];
const genres = ['Grunge', 'Rock', 'Industrial', 'Boys Band', 'Dance', 'Techno'];

const user_votes = tf.tensor([
    [10,3,1,6,3,2,1],
    [0,2,6,10,2,4,2],
    [1,6,2,9,1,7,10],
    [6,5,1,1,1,2,9],
]);

const band_genres = tf.tensor([
    [1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1],
    [1, 1, 0, 0, 0, 0],
]);

const user_genres = tf.matMul(user_votes, band_genres);
user_genres.print();

const top_user_genres = tf.topk(user_genres, genres.length);
const top_genres = top_user_genres.indices.arraySync();

users.forEach((user, i) => {
    console.log(user, top_genres[i].map((v: number) => genres[v]))
})
